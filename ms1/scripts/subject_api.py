import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from string import Formatter
from typing import Dict, Optional, Tuple
import re


_LOG = logging.getLogger(__name__)
BASE_ROOT = Path(os.environ.get("MS_ROOT", Path(__file__).resolve().parents[1]))
class _SafeDict(dict):
    def __missing__(self, key):  # type: ignore[override]
        return "{" + key + "}"


def _safe_format(template: str, mapping: Dict[str, str]) -> str:
    fmt = Formatter()
    return fmt.vformat(template, (), _SafeDict(mapping))


def _read_mesh(mesh_path: str) -> Tuple[list, list]:
    verts = []
    faces = []
    with open(mesh_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        verts.append(tuple(float(p) for p in parts[1:4]))
                    except ValueError:
                        continue
            elif line.startswith("f "):
                parts = line.strip().split()
                indices = []
                for p in parts[1:]:
                    try:
                        indices.append(int(p.split("/")[0]) - 1)
                    except ValueError:
                        continue
                if len(indices) >= 3:
                    faces.append(indices[:3])
    if not verts:
        raise ValueError("no vertices parsed from mesh")
    return verts, faces


def _sample_points(verts: list, num_points: int = 2048) -> list:
    if not verts:
        raise ValueError("no vertices provided for sampling")
    return [random.choice(verts) for _ in range(num_points)]


def _write_point_cloud(points, out_path: str) -> None:
    with open(out_path, "w") as f:
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


def _canonical(value: Optional[str]) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


class SubjectRunner:
    """Wraps a subject for single-mesh inference with entry templates and adapters."""

    def __init__(
        self,
        spec: Dict,
        dataset_root: Optional[str],
        entry_name: str,
        entry_template: str,
        device: str,
        weights_path: Optional[str] = None,
        extra: Optional[Dict] = None,
        workdir: Optional[str] = None,
        pre_cmd: Optional[str] = None,
        dataset_name: Optional[str] = None,
        auto_append_mesh: Optional[bool] = None,
    ):
        self.spec = spec
        self.dataset_root = dataset_root or ""
        self.entry_name = entry_name
        self.entry_template = entry_template
        self.device = device or ""
        self.weights_path = weights_path or ""
        self.extra = extra or {}
        raw_workdir = workdir or spec.get("workdir")
        if raw_workdir:
            candidate = Path(raw_workdir)
            if not candidate.is_absolute():
                candidate = BASE_ROOT / candidate
            candidate.mkdir(parents=True, exist_ok=True)
            self.workdir = str(candidate)
        else:
            self.workdir = None
        self.pre_cmd = pre_cmd or spec.get("pre_cmd")
        if self.pre_cmd:
            env_match = re.search(r"conda activate\s+([\w\-]+)", self.pre_cmd)
            if env_match:
                env_name = env_match.group(1)
                candidate_paths = [
                    Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python",
                    Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "python"
                    if os.environ.get("CONDA_PREFIX") else None,
                ]
                candidate_python = next((p for p in candidate_paths if p and p.exists()), None)
                if candidate_python:
                    pattern = "python -c \"import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))\""
                    replacement = f"{candidate_python} -c \"import os,torch;print(os.path.join(os.path.dirname(torch.__file__),'lib'))\""
                    self.pre_cmd = self.pre_cmd.replace(pattern, replacement)
        self.input_format = spec.get("input_format", "triangle_mesh")
        self.adapter = spec.get("adapter") or spec.get("input_adapter", self.input_format)
        self.dataset_name = dataset_name or ""
        if auto_append_mesh is None:
            auto_append_mesh = spec.get("auto_append_mesh", True)
        self.auto_append_mesh = bool(auto_append_mesh)
        exec_path = spec.get("path")
        if exec_path:
            candidate = Path(exec_path)
            if not candidate.is_absolute():
                candidate = BASE_ROOT / candidate
            self.exec_dir = str(candidate)
        else:
            self.exec_dir = None

    def _prepare_input(self, mesh_path: str) -> Tuple[str, Optional[str]]:
        """Return path passed to command and temp dir to cleanup."""
        adapter = (self.adapter or "triangle_mesh").lower()
        if adapter in ("triangle_mesh", "mesh"):
            return mesh_path, None
        tmpdir = tempfile.mkdtemp(prefix="ms1_subject_")
        try:
            if adapter in ("point_cloud", "mesh_to_point", "point"):
                verts, _ = _read_mesh(mesh_path)
                points = _sample_points(verts)
                out = os.path.join(tmpdir, Path(mesh_path).stem + ".xyz")
                _write_point_cloud(points, out)
                return out, tmpdir
            elif adapter in ("graph", "mesh_to_graph"):
                # Trivial adjacency derived from faces
                verts, faces = _read_mesh(mesh_path)
                out = os.path.join(tmpdir, Path(mesh_path).stem + ".json")
                nodes = [list(pt) for pt in verts]
                edges = set()
                for face in faces:
                    a, b, c = face[:3]
                    edges.add(tuple(sorted((a, b))))
                    edges.add(tuple(sorted((b, c))))
                    edges.add(tuple(sorted((a, c))))
                data = {"nodes": nodes, "edges": [[a, b] for (a, b) in edges]}
                with open(out, "w") as f:
                    json.dump(data, f)
                return out, tmpdir
            else:
                _LOG.warning("Unsupported adapter '%s', falling back to original mesh", adapter)
                shutil.rmtree(tmpdir, ignore_errors=True)
                return mesh_path, None
        except Exception:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise

    def _build_command(self, mesh_path: str) -> Tuple[str, Dict[str, str]]:
        command_inputs = {
            "mesh": mesh_path,
            "device": self.device,
            "weights": self.weights_path,
            "data_root": self.dataset_root,
            "weights_path": self.weights_path,
        }
        command_inputs.update({k: str(v) for k, v in self.extra.items()})
        cmd = _safe_format(self.entry_template, command_inputs)
        dataset_key = _canonical(self.dataset_name)

        def replace_dataset(match):
            key = _canonical(match.group(1))
            if key == dataset_key and self.dataset_root:
                return self.dataset_root
            return match.group(0)

        cmd = re.sub(r"\$\{DATASET:([^}]+)\}", replace_dataset, cmd)
        if self.auto_append_mesh and mesh_path and "{mesh}" not in self.entry_template:
            cmd = f"{cmd} {shlex_quote(mesh_path)}"
        if self.pre_cmd:
            preamble = self.pre_cmd.strip()
            if preamble:
                cmd = f"{preamble} && {cmd}"
        return cmd, command_inputs

    def run_once(self, mesh_path: str, timeout_sec: int) -> Dict:
        self.last_command = None
        self.last_inputs = None

        overall_start = time.time()
        try:
            input_path, tmpdir = self._prepare_input(mesh_path)
        except Exception:
            end = time.time()
            _LOG.debug(
                "Input preparation failed for subject '%s'", self.spec.get("name"), exc_info=True
            )
            return {
                "accepted": False,
                "crashed": False,
                "error_type": "parse_error",
                "prediction": None,
                "time_sec": float(end - overall_start),
                "command": None,
                "inputs": None,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "start_time": overall_start,
                "end_time": end,
            }
        try:
            cmd, mapping = self._build_command(input_path)
        except Exception as exc:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
            raise exc

        env = os.environ.copy()
        if self.device:
            env.setdefault("MS1_DEVICE", self.device)
        if self.weights_path:
            env.setdefault("MS1_WEIGHTS", self.weights_path)
        if self.dataset_root:
            env.setdefault("MS1_DATA_ROOT", self.dataset_root)
        if self.workdir:
            env.setdefault("MS1_WORKDIR", self.workdir)

        cwd = self.exec_dir or self.workdir or os.getcwd()
        exec_start = time.time()
        try:
            _LOG.debug("[%s] running: %s", self.spec.get("name"), cmd)
            proc = subprocess.run(
                ["/bin/bash", "-lc", cmd],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                env=env,
                cwd=cwd,
                text=True,
            )
            end = time.time()
            dur = end - exec_start
            accepted = proc.returncode == 0
            error_type: Optional[str] = None
            crashed = False
            if proc.returncode != 0:
                stderr_lower = (proc.stderr or "").lower()
                if "parse" in stderr_lower or "invalid" in stderr_lower:
                    error_type = "parse_error"
                else:
                    error_type = "runtime_error"
            prediction = None
            output = (proc.stdout or "").strip()
            if output:
                for line in output.splitlines():
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "prediction" in obj:
                            pred_val = obj.get("prediction")
                            prediction = None if pred_val is None else str(pred_val)
                            break
                    except json.JSONDecodeError:
                        continue
            result = {
                "accepted": bool(accepted),
                "crashed": bool(crashed),
                "error_type": error_type,
                "prediction": prediction,
                "time_sec": float(dur),
                "command": cmd,
                "inputs": mapping,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "exit_code": proc.returncode,
                "start_time": exec_start,
                "end_time": end,
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        except subprocess.TimeoutExpired as exc:
            end = time.time()
            result = {
                "accepted": False,
                "crashed": False,
                "error_type": "timeout",
                "prediction": None,
                "time_sec": float(end - exec_start),
                "command": cmd,
                "inputs": mapping,
                "stdout": exc.stdout or "",
                "stderr": exc.stderr or "",
                "exit_code": -2,
                "start_time": exec_start,
                "end_time": end,
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        except FileNotFoundError:
            end = time.time()
            _LOG.warning("Entry command not found for subject '%s'", self.spec.get("name"))
            result = {
                "accepted": False,
                "crashed": False,
                "error_type": "runtime_error",
                "prediction": None,
                "time_sec": float(end - exec_start),
                "command": cmd,
                "inputs": mapping,
                "stdout": "",
                "stderr": "",
                "exit_code": -1,
                "start_time": exec_start,
                "end_time": end,
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        except Exception as exc:
            end = time.time()
            _LOG.exception("Subject '%s' crashed: %s", self.spec.get("name"), exc)
            result = {
                "accepted": False,
                "crashed": True,
                "error_type": "runtime_error",
                "prediction": None,
                "time_sec": float(end - exec_start),
                "command": cmd,
                "inputs": mapping,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -3,
                "start_time": exec_start,
                "end_time": end,
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)


def shlex_quote(value: str) -> str:
    if not value:
        return "''"
    if all(ch.isalnum() or ch in "@%_-+=:,./" for ch in value):
        return value
    return "'" + value.replace("'", "'\\''") + "'"
