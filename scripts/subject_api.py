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


_LOG = logging.getLogger(__name__)
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
    ):
        self.spec = spec
        self.dataset_root = dataset_root or ""
        self.entry_name = entry_name
        self.entry_template = entry_template
        self.device = device or ""
        self.weights_path = weights_path or ""
        self.extra = extra or {}
        self.workdir = spec.get("path")
        self.input_format = spec.get("input_format", "triangle_mesh")
        self.adapter = spec.get("input_adapter", self.input_format)

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
        }
        command_inputs.update({k: str(v) for k, v in self.extra.items()})
        cmd = _safe_format(self.entry_template, command_inputs)
        if "{mesh}" not in self.entry_template:
            cmd = f"{cmd} {shlex_quote(mesh_path)}"
        return cmd, command_inputs

    def run_once(self, mesh_path: str, timeout_sec: int) -> Dict:
        self.last_command: Optional[str] = None
        self.last_inputs: Optional[Dict[str, str]] = None

        start = time.time()
        try:
            input_path, tmpdir = self._prepare_input(mesh_path)
        except Exception:
            dur = time.time() - start
            _LOG.debug(
                "Input preparation failed for subject '%s'", self.spec.get("name"), exc_info=True
            )
            return {
                "accepted": False,
                "crashed": False,
                "error_type": "parse_error",
                "prediction": None,
                "time_sec": float(dur),
                "command": None,
                "inputs": None,
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

        cwd = self.workdir or os.getcwd()
        start = time.time()
        try:
            _LOG.debug("[%s] running: %s", self.spec.get("name"), cmd)
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                env=env,
                cwd=cwd,
                text=True,
            )
            dur = time.time() - start
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
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        except subprocess.TimeoutExpired:
            dur = time.time() - start
            result = {
                "accepted": False,
                "crashed": False,
                "error_type": "timeout",
                "prediction": None,
                "time_sec": float(dur),
                "command": cmd,
                "inputs": mapping,
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        except FileNotFoundError:
            dur = time.time() - start
            _LOG.warning("Entry command not found for subject '%s'", self.spec.get("name"))
            result = {
                "accepted": False,
                "crashed": False,
                "error_type": "runtime_error",
                "prediction": None,
                "time_sec": float(dur),
                "command": cmd,
                "inputs": mapping,
            }
            self.last_command = cmd
            self.last_inputs = mapping
            return result
        except Exception as exc:
            dur = time.time() - start
            _LOG.exception("Subject '%s' crashed: %s", self.spec.get("name"), exc)
            result = {
                "accepted": False,
                "crashed": True,
                "error_type": "runtime_error",
                "prediction": None,
                "time_sec": float(dur),
                "command": cmd,
                "inputs": mapping,
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
