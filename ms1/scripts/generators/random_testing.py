import logging
import os
import random
import shutil
import tempfile
import time
from typing import Dict, Tuple

try:
    from ..subject_api import SubjectRunner
except ImportError:  # pragma: no cover - fallback when executed as a script
    from subject_api import SubjectRunner  # type: ignore

_LOG = logging.getLogger(__name__)


def _perturb_seed(src: str, dst: str) -> None:
    """Light perturbation: copy and append a noop OBJ comment to keep it well-formed."""
    with open(src, "r", errors="ignore") as f_in, open(dst, "w") as f_out:
        f_out.write(f_in.read())
        f_out.write(f"\n# ms1_random_perturb_{random.randint(0, 1_000_000)}\n")


def run(
    subject: SubjectRunner,
    seed_path: str,
    time_budget_sec: int,
    threads: int,
    mutation_budget: int,
) -> Tuple[Dict, Dict]:
    """
    Executes random testing by repeatedly perturbing the seed and invoking the subject.
    Returns (metrics, meta) where metrics match the JSONL schema fields subset, and meta has tool metadata.
    """
    start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="ms1_rand_")
    generated_total = 0
    valid_count = 0
    invalid_breakdown = {"parse_error": 0, "runtime_error": 0, "timeout": 0, "constraint_fail": 0}
    crash_count = 0

    try:
        for i in range(mutation_budget):
            if (time.time() - start) >= time_budget_sec:
                break
            dst = os.path.join(tmpdir, f"mut_{i}.obj")
            try:
                _perturb_seed(seed_path, dst)
            except Exception:
                # treat as parse error of generated
                invalid_breakdown["parse_error"] += 1
                generated_total += 1
                continue
            res = subject.run_once(dst, timeout_sec=max(1, int(min(30, time_budget_sec))))
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
    meta = {
        "tool_version": "random_testing v0",
        "tool_args": f"--threads={threads} --budget={mutation_budget}",
    }
    return metrics, meta
