#!/usr/bin/env python
"""Generic AFL harness wrapper used for MS1/MS2 fuzzing."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import trimesh


SUBJECT_CMD_TEMPLATES = {
    # Placeholder commands – swap in real inference harnesses when available.
    # Example: "python scripts/point2mesh_infer.py --input {input} --output {output}"
    "Point2Mesh": None,
    "MeshCNN": None,
    "HodgeNet": "python scripts/hodgenet_infer.py --input {input} --data-root data/datasets/ModelNet40_lite600",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AFL harness wrapper for MS1 subjects")
    parser.add_argument("subject", help="Subject name (e.g., Point2Mesh, MeshCNN, HodgeNet)")
    parser.add_argument("input", nargs="?", help="Input file (mesh) path")
    parser.add_argument("--input", dest="input_flag", help="Input file path (alternative)")
    return parser.parse_args()


def validate_mesh(path: Path) -> bool:
    mesh = trimesh.load(path, force="mesh")
    if mesh.is_empty:
        return False
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()
    return len(mesh.vertices) > 3 and len(mesh.faces) > 0


def run_command(template: str, input_path: Path) -> int:
    with tempfile.NamedTemporaryFile(suffix="_afl_out", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        cmd = template.format(input=str(input_path), output=str(out_path))
        result = subprocess.run(cmd, shell=True, check=False)
        return result.returncode
    finally:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass


def main() -> int:
    args = parse_args()
    input_arg = args.input_flag or args.input
    if not input_arg:
        print(json.dumps({"error": "no input provided"}), flush=True)
        return 1

    subject = args.subject
    input_path = Path(input_arg)
    start = time.time()

    template = SUBJECT_CMD_TEMPLATES.get(subject)
    if template:
        exit_code = run_command(template, input_path)
        valid = exit_code == 0
    else:
        valid = validate_mesh(input_path)
        exit_code = 0 if valid else 1

    duration = time.time() - start
    summary = {
        "subject": subject,
        "input": str(input_path),
        "exit_code": exit_code,
        "valid": bool(valid),
        "duration_s": duration,
    }
    print(json.dumps(summary), flush=True)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
