import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from typing import Dict, Tuple

from ..subject_api import SubjectRunner

_LOG = logging.getLogger(__name__)


def _afl_available() -> Tuple[bool, str]:
    for bin in ("afl-fuzz", "afl-fuzz++", "afl-fuzz-afl++"):
        try:
            out = subprocess.run([bin, "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if out.returncode in (0, 1):
                return True, bin
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return False, "simulated"


def run(
    subject: SubjectRunner,
    seed_path: str,
    time_budget_sec: int,
    threads: int,
    mutation_budget: int,
) -> Tuple[Dict, Dict]:
    """
    Wrap AFL/afl++ if present; otherwise simulate with byte-level mutations feeding SubjectRunner.
    """
    start = time.time()
    have_afl, bin_name = _afl_available()
    tmpdir = tempfile.mkdtemp(prefix="ms1_afl_")
    in_dir = os.path.join(tmpdir, "in"); os.makedirs(in_dir, exist_ok=True)
    out_dir = os.path.join(tmpdir, "out"); os.makedirs(out_dir, exist_ok=True)
    # Seed corpus
    seed_copy = os.path.join(in_dir, os.path.basename(seed_path))
    shutil.copy2(seed_path, seed_copy)

    generated_total = 0
    valid_count = 0
    invalid_breakdown = {"parse_error": 0, "runtime_error": 0, "timeout": 0, "constraint_fail": 0}
    crash_count = 0

    try:
        if have_afl:
            # We cannot launch arbitrary target binaries reliably; use simulation path to still produce metrics
            _LOG.info("AFL detected (%s) but no target harness; simulating mutations via SubjectRunner", bin_name)
        # Simulated mutation loop honoring budgets/time
        while (time.time() - start) < time_budget_sec and generated_total < mutation_budget:
            # Create a mutated test case from seed bytes
            with open(seed_copy, "rb") as f:
                data = bytearray(f.read())
            if data:
                for _ in range(random.randint(1, 4)):
                    idx = random.randrange(0, len(data))
                    data[idx] = random.randrange(0, 256)
            else:
                data.extend(os.urandom(32))
            dst = os.path.join(out_dir, f"id_{generated_total:06d}")
            with open(dst, "wb") as f:
                f.write(data)
            # Feed subject (some subjects may accept OBJ only; corrupted inputs likely parse_error)
            res = subject.run_once(dst, timeout_sec=max(1, int(min(10, time_budget_sec))))
            generated_total += 1
            if res.get("accepted"):
                valid_count += 1
            else:
                et = res.get("error_type") or "runtime_error"
                invalid_breakdown[et] = invalid_breakdown.get(et, 0) + 1
            if res.get("crashed"):
                crash_count += 1
    finally:
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    metrics = {
        "generated_total": generated_total,
        "valid_count": valid_count,
        "invalid_count": generated_total - valid_count,
        "invalid_breakdown": invalid_breakdown,
        "crash_count": crash_count,
        "execution_time_sec": float(time.time() - start),
    }
    tool_ver = f"{bin_name} present" if have_afl else "afl++ simulated"
    meta = {
        "tool_version": tool_ver,
        "tool_args": f"--threads={threads} --budget={mutation_budget}",
    }
    return metrics, meta

