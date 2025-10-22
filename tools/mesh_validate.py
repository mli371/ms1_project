#!/usr/bin/env python
"""Simple mesh validation/repair utility for MS1/MS2 pipelines."""

import argparse
import json
import sys
from pathlib import Path

import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and optionally repair a mesh")
    parser.add_argument("mesh_path", type=Path, help="Path to the input mesh (OBJ/PLY/etc.)")
    parser.add_argument("--repair-out", type=Path, dest="repair_out", help="Optional path to write the repaired mesh")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mesh = trimesh.load(args.mesh_path, force="mesh")
    if mesh.is_empty:
        summary = {"path": str(args.mesh_path), "valid": False, "n_verts": 0, "n_faces": 0}
        print(json.dumps(summary), flush=True)
        print("bad", flush=True)
        return 1

    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.fix_normals()

    n_verts = int(len(mesh.vertices))
    n_faces = int(len(mesh.faces))
    valid = n_faces > 0 and n_verts > 3

    summary = {
        "path": str(args.mesh_path),
        "valid": bool(valid),
        "n_verts": n_verts,
        "n_faces": n_faces,
    }
    print(json.dumps(summary), flush=True)

    if args.repair_out is not None:
        mesh.export(args.repair_out)

    print("ok" if valid else "bad", flush=True)
    return 0 if valid else 1


if __name__ == "__main__":
    sys.exit(main())
