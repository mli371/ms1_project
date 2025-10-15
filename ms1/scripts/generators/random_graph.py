import logging
import os
import random
import shutil
import tempfile
import time
from typing import Dict, Tuple

try:
    from ..subject_api import SubjectRunner
except ImportError:  # pragma: no cover
    from subject_api import SubjectRunner  # type: ignore

_LOG = logging.getLogger(__name__)


def _erdos_renyi_graph(n: int, p: float):
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                edges.append((i, j))
    return edges


def _graph_to_triangle_fan_obj(n: int, edges, path: str):
    # Simple layout on circle -> triangle fan around vertex 0
    import math
    verts = []
    for k in range(n):
        angle = 2 * math.pi * k / max(1, n)
        verts.append((math.cos(angle), math.sin(angle), 0.0))
    with open(path, "w") as f:
        for x, y, z in verts:
            f.write(f"v {x} {y} {z}\n")
        # Use any two consecutive neighbors of 0 to form triangles
        nbrs = sorted([b if a == 0 else a for (a, b) in edges if a == 0 or b == 0])
        for i in range(1, len(nbrs) - 1):
            f.write(f"f {1} {nbrs[i] + 1} {nbrs[i + 1] + 1}\n")


def run(
    subject: SubjectRunner,
    seed_path: str,
    time_budget_sec: int,
    threads: int,
    mutation_budget: int,
) -> Tuple[Dict, Dict]:
    start = time.time()
    tmpdir = tempfile.mkdtemp(prefix="ms1_randgraph_")
    generated_total = 0
    valid_count = 0
    invalid_breakdown = {"parse_error": 0, "runtime_error": 0, "timeout": 0, "constraint_fail": 0}
    crash_count = 0
    t_graph_build = 0.0

    try:
        for i in range(mutation_budget):
            if (time.time() - start) >= time_budget_sec:
                break
            dst = os.path.join(tmpdir, f"graph_{i}.obj")
            t0 = time.time()
            try:
                n = random.randint(8, 32)
                p = random.uniform(0.1, 0.3)
                edges = _erdos_renyi_graph(n, p)
                _graph_to_triangle_fan_obj(n, edges, dst)
            except Exception:
                invalid_breakdown["parse_error"] += 1
                generated_total += 1
                continue
            t_graph_build += time.time() - t0
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
        "t_graph_build": float(t_graph_build),
    }
    meta = {
        "tool_version": "random_graph v0",
        "tool_args": f"--threads={threads} --budget={mutation_budget}",
    }
    return metrics, meta
