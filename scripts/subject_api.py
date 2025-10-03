import json
import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional, Dict


_LOG = logging.getLogger(__name__)


@dataclass
class SubjectSpec:
    name: str
    entry_cmd: str
    device: str


class SubjectRunner:
    def __init__(self, name: str, entry_cmd: str, device: str):
        self.spec = SubjectSpec(name=name, entry_cmd=entry_cmd, device=device)

    def run_once(self, mesh_path: str, timeout_sec: int) -> Dict:
        """
        Execute the subject's CLI. The orchestrator appends the input path as the last arg.

        Returns dict with keys:
         - accepted: bool
         - crashed: bool
         - error_type: Optional[str]
         - prediction: Optional[str]
         - time_sec: float
        """
        cmd = f"{self.spec.entry_cmd} {shlex.quote(mesh_path)}"
        env = os.environ.copy()
        if self.spec.device:
            env["MS1_DEVICE"] = self.spec.device
        start = time.time()
        try:
            _LOG.debug("Running subject '%s': %s", self.spec.name, cmd)
            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_sec,
                env=env,
                text=True,
            )
            dur = time.time() - start
            accepted = proc.returncode == 0
            crashed = False
            error_type: Optional[str] = None
            if proc.returncode != 0:
                # Heuristics to categorize error types
                stderr = (proc.stderr or "").lower()
                if "parse" in stderr or "syntax" in stderr:
                    error_type = "parse_error"
                else:
                    error_type = "runtime_error"
            # Attempt to parse prediction from stdout if JSON line present
            prediction = None
            out = (proc.stdout or "").strip()
            if out:
                for line in out.splitlines():
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "prediction" in obj:
                            prediction = str(obj["prediction"]) if obj["prediction"] is not None else None
                            break
                    except Exception:
                        continue
            return {
                "accepted": bool(accepted),
                "crashed": bool(crashed),
                "error_type": error_type,
                "prediction": prediction,
                "time_sec": float(dur),
            }
        except subprocess.TimeoutExpired:
            dur = time.time() - start
            return {
                "accepted": False,
                "crashed": False,
                "error_type": "timeout",
                "prediction": None,
                "time_sec": float(dur),
            }
        except FileNotFoundError:
            dur = time.time() - start
            _LOG.warning("Entry command not found for subject '%s' (simulating runtime_error)", self.spec.name)
            return {
                "accepted": False,
                "crashed": False,
                "error_type": "runtime_error",
                "prediction": None,
                "time_sec": float(dur),
            }
        except Exception as e:
            dur = time.time() - start
            _LOG.exception("Subject '%s' crashed: %s", self.spec.name, e)
            return {
                "accepted": False,
                "crashed": True,
                "error_type": "runtime_error",
                "prediction": None,
                "time_sec": float(dur),
            }

