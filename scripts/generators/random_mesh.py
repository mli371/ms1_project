import logging
import os
import random
import shutil
import tempfile
import time
from typing import Dict, Tuple

from ..subject_api import SubjectRunner

_LOG = logging.getLogger(__name__)


def _write_cube_obj(path: str, noise: float = 0.0) -> None:
    verts = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    if noise:
        verts = [(x + random.uniform(-noise, noise), y + random.uniform(-noise, noise), z + random.uniform(-noise, noise)) for x, y, z in verts]
    faces = [
        (1, 2, 3), (1, 3, 4),  # bottom
        (5, 6, 7), (5, 7, 8),  # top
        (1, 2, 6), (1, 6, 5),  # side
        (2, 3, 7), (2, 7, 6),
        (3, 4, 8), (3, 8, 7),
        (4, 1, 5), (4, 5, 8),
    ]
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        for a, b, c in faces:
            f.write(f"f {a} {b} {c}\n")


def run(
    subject: SubjectRunner,
    seed_path: str,
    time_budget_sec: int,
    threads: int,
    mutation_budget: int,
) -> Tuple[Dict, Dict]:
    start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="ms1_randmesh_")
    generated_total = 0
    valid_count = 0
    invalid_breakdown = {"parse_error": 0, "runtime_error": 0, "timeout": 0, "constraint_fail": 0}
    crash_count = 0

    try:
        for i in range(mutation_budget):
            if (time.time() - start) >= time_budget_sec:
                break
            dst = os.path.join(tmpdir, f"mesh_{i}.obj")
            try:
                _write_cube_obj(dst, noise=min(0.2, random.random() * 0.1))
            except Exception:
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
        "tool_version": "random_mesh v0",
        "tool_args": f"--threads={threads} --budget={mutation_budget}",
    }
    return metrics, meta

