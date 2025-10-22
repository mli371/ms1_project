"""Random mesh generator utilities for MS1/MS2 baselines.

This module keeps the existing Python API used by the MS1 runner (`run`) and
adds a standalone CLI for producing seed meshes on disk.
"""

import argparse
import json
import logging
import os
import random
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Tuple, TYPE_CHECKING

import numpy as np
import trimesh

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from ..subject_api import SubjectRunner
else:  # CLI execution path does not need the runner available
    SubjectRunner = Any  # type: ignore


_LOG = logging.getLogger(__name__)


def _generate_random_points(num_points: int) -> np.ndarray:
    """Sample points roughly on a perturbed sphere."""

    pts = np.random.normal(size=(num_points, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    radii = np.random.uniform(0.7, 1.3, size=(num_points, 1))
    pts *= radii
    return pts


def _convex_hull_mesh(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return vertices/faces for the convex hull of the sampled points."""

    hull = trimesh.points.PointCloud(points).convex_hull
    return hull.vertices, hull.faces


def generate_random_mesh(min_verts: int, max_verts: int, max_attempts: int = 6) -> trimesh.Trimesh:
    """Create a random mesh whose vertex count falls within the requested bounds."""

    for attempt in range(max_attempts):
        target = random.randint(min_verts, max_verts)
        pts = _generate_random_points(max(target * 2, target + 10))
        vertices, faces = _convex_hull_mesh(pts)
        if min_verts <= len(vertices) <= max_verts and len(faces) > 0:
            break
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def validate_and_fix(mesh: trimesh.Trimesh) -> Tuple[trimesh.Trimesh, bool]:
    """Clean up degeneracies, duplicates, and normals before exporting."""

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    mesh.rezero()
    mesh.remove_infinite_values()
    is_valid = mesh.is_watertight and len(mesh.vertices) > 0 and len(mesh.faces) > 0
    return mesh, is_valid


def run(
    subject: SubjectRunner,
    seed_path: str,
    time_budget_sec: int,
    threads: int,
    mutation_budget: int,
) -> Tuple[Dict, Dict]:
    """Legacy MS1 entry point: generate meshes in a temp directory and fuzz the subject."""

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
                mesh, _ = validate_and_fix(generate_random_mesh(20, 60))
                mesh.export(dst)
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
        "tool_version": "random_mesh v1",
        "tool_args": f"--threads={threads} --budget={mutation_budget}",
    }
    return metrics, meta


def generate_dataset(out_dir: Path, count: int, min_verts: int, max_verts: int) -> None:
    """Produce a dataset of random meshes and JSON metadata in the given directory."""

    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        mesh = generate_random_mesh(min_verts, max_verts)
        mesh, valid = validate_and_fix(mesh)
        mesh_path = out_dir / f"rand_mesh_{idx:04d}.obj"
        mesh.export(mesh_path)
        log_path = mesh_path.with_suffix(".json")
        with open(log_path, "w") as f:
            json.dump(
                {
                    "valid": bool(valid),
                    "n_verts": int(len(mesh.vertices)),
                    "n_faces": int(len(mesh.faces)),
                },
                f,
                indent=2,
            )
        print(
            f"[{idx + 1}/{count}] {mesh_path.name} valid={valid} "
            f"verts={len(mesh.vertices)} faces={len(mesh.faces)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random OBJ meshes for fuzzing seeds")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/seeds/mesh"),
        help="Directory to store generated meshes",
    )
    parser.add_argument("--count", type=int, default=100, help="Number of meshes to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    parser.add_argument("--min_verts", type=int, default=50, help="Minimum number of vertices per mesh")
    parser.add_argument("--max_verts", type=int, default=500, help="Maximum number of vertices per mesh")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    generate_dataset(args.out_dir, args.count, args.min_verts, args.max_verts)


if __name__ == "__main__":
    main()
