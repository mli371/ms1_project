#!/usr/bin/env python
"""End-to-end baseline orchestrator for MS1 fuzz/seed workflows.

Steps (per subject):
1. Generate seed meshes into `data/seeds/{subject}` (if requested).
2. Validate seeds with `tools/mesh_validate.py`.
3. Replay/benchmark using `python -m scripts.ms1_runner --seed-dir ... --subject ...`.
4. Append summary CSVs under `workdir/baseline_runs/`.

The script is designed so additional subjects (e.g., MeshCNN, MeshWalker) can
be plugged in later by extending `SUBJECT_CONFIG` with generator/runner config.
"""

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]

PYTHON = sys.executable

SUBJECT_CONFIG = {
    "Point2Mesh": {
        "env": "point2mesh-gpu",
        "seed_dir": REPO_ROOT / "data/seeds/point2meshSeeds",
        "generator": [
            "python",
            str(REPO_ROOT / "scripts/generators/random_mesh.py"),
            "--count",
            "100",
            "--min_verts",
            "150",
            "--max_verts",
            "350",
            "--out-dir",
            str(REPO_ROOT / "data/seeds/point2meshSeeds"),
        ],
        "runner_args": [
            "--seed-dir",
            str(REPO_ROOT / "data/seeds/point2meshSeeds"),
            "--subject",
            "Point2Mesh",
            "--parallel",
            "2",
        ],
    },
    "HodgeNet": {
        "env": "HodgeNet",
        "seed_dir": REPO_ROOT / "data/seeds/hodgenetSeeds",
        "generator": None,  # Placeholder: currently reuse curated dataset seeds
        "runner_args": [
            "--seed-dir",
            str(REPO_ROOT / "data/seeds/hodgenetSeeds"),
            "--subject",
            "HodgeNet",
            "--parallel",
            "1",
        ],
    },
    # Future subjects – placeholders for quick integration
    "MeshCNN": {
        "env": "meshcnn",
        "seed_dir": REPO_ROOT / "data/seeds/meshcnnSeeds",
        "generator": None,
        "runner_args": [
            "--seed-dir",
            str(REPO_ROOT / "data/seeds/meshcnnSeeds"),
            "--subject",
            "MeshCNN",
            "--parallel",
            "1",
        ],
    },
    "MeshWalker": {
        "env": "meshwalker",
        "seed_dir": REPO_ROOT / "data/seeds/meshwSeeds",
        "generator": None,
        "runner_args": [
            "--seed-dir",
            str(REPO_ROOT / "data/seeds/meshwSeeds"),
            "--subject",
            "MeshWalker",
            "--parallel",
            "1",
        ],
    },
}


def conda_run(env: str, argv: List[str]) -> subprocess.CompletedProcess:
    full_cmd = ["conda", "run", "--no-capture-output", "-n", env] + argv
    return subprocess.run(full_cmd, check=False)


def ensure_seed_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def maybe_generate(subject: str, cfg: Dict[str, Optional[List[str]]]) -> None:
    generator_cmd = cfg.get("generator")
    if not generator_cmd:
        return
    seed_dir = cfg["seed_dir"]
    ensure_seed_dir(seed_dir)
    env = cfg["env"]
    print(f"[{subject}] Generating seeds via {generator_cmd}")
    start = time.time()
    result = conda_run(env, generator_cmd)
    if result.returncode != 0:
        print(f"[{subject}] Seed generation failed (exit={result.returncode})")
    else:
        print(f"[{subject}] Seed generation completed in {time.time() - start:.1f}s")


def run_ms1_runner(subject: str, cfg: Dict[str, Optional[List[str]]], out_csv: Path) -> Path:
    env = cfg["env"]
    seed_dir = cfg["seed_dir"]
    ensure_seed_dir(seed_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "-m",
        "scripts.ms1_runner",
        *cfg["runner_args"],
        "--out-csv",
        str(out_csv),
    ]
    print(f"[{subject}] Running ms1_runner: {' '.join(command)}")
    start = time.time()
    result = conda_run(env, command)
    if result.returncode != 0:
        print(f"[{subject}] ms1_runner failed (exit={result.returncode})")
    else:
        print(f"[{subject}] ms1_runner completed in {time.time() - start:.1f}s")
    return out_csv


def summarize_csv(path: Path) -> Dict[str, int | float]:
    total = 0
    valid = 0
    failed = 0
    duration_sum = 0.0
    if not path.exists():
        return {"total": 0, "valid": 0, "failed": 0, "mean_duration": 0.0}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            valid += int(row.get("valid", "0") in ("True", "1"))
            failed += int(row.get("exit_code", "1") != "0")
            duration_sum += float(row.get("duration_s", "0.0"))
    mean_duration = duration_sum / max(1, total)
    return {"total": total, "valid": valid, "failed": failed, "mean_duration": mean_duration}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline pipeline across subjects")
    parser.add_argument("--subjects", nargs="+", default=["Point2Mesh", "HodgeNet"], help="Subjects to process")
    parser.add_argument("--skip-generate", action="store_true", help="Skip random seed generation step")
    parser.add_argument("--out-dir", type=Path, default=Path("workdir/baseline_runs"), help="Output directory for CSV summary")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary: Dict[str, Dict[str, int | float]] = {}
    for subject in args.subjects:
        cfg = SUBJECT_CONFIG.get(subject)
        if not cfg:
            print(f"[WARN] Subject {subject} not configured; skipping")
            continue
        if not args.skip_generate:
            maybe_generate(subject, cfg)
        out_csv = args.out_dir / f"{subject.lower()}_baseline.csv"
        run_ms1_runner(subject, cfg, out_csv)
        summary[subject] = summarize_csv(out_csv)

    print("=== Baseline Summary ===")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
