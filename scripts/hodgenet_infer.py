#!/usr/bin/env python
"""Single-mesh inference utility for HodgeNet classifiers."""

import argparse
import json
import sys
# from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional

if __package__ is None or __package__ == "":  # pragma: no cover
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    hodgenet_root = repo_root / "subjects_src" / "HodgeNet"
    if str(hodgenet_root) not in sys.path:
        sys.path.insert(0, str(hodgenet_root))

import torch
import torch.nn as nn

from subjects_src.HodgeNet.hodgenet import HodgeNetModel
from subjects_src.HodgeNet.meshdata import HodgenetMeshDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HodgeNet inference on a single mesh")
    parser.add_argument("--input", required=True, type=Path, help="Path to the input mesh")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("workdir/ModelNet40/HodgeNet_lite/baseline_20/best.pth"),
        help="Checkpoint to load (default: baseline_20/best.pth)",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu)")
    parser.add_argument("--data-root", type=Path, help="Optional path to data root for label mapping")
    parser.add_argument("--out", type=Path, help="Optional path to write JSON output")
    parser.add_argument("--n-eig", type=int, default=32)
    parser.add_argument("--n-extra-eig", type=int, default=32)
    parser.add_argument("--n-out-features", type=int, default=32)
    parser.add_argument("--num-vector-dimensions", type=int, default=4)
    return parser.parse_args()


def load_label_map(data_root: Optional[Path]) -> Optional[Dict[int, str]]:
    if not data_root:
        return None
    labels_file = data_root / "labels.txt"
    if not labels_file.exists():
        return None
    mapping: Dict[int, str] = {}
    with labels_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            rel_path, label_idx = parts
            try:
                idx = int(label_idx)
            except ValueError:
                continue
            parts_path = Path(rel_path).parts
            if len(parts_path) >= 2:
                name = parts_path[1]
            else:
                name = parts_path[0]
            mapping.setdefault(idx, name)
    return mapping or None


def move_batch_to_device(batch: list[dict], target_device: torch.device) -> list[dict]:
    moved = []
    for mesh in batch:
        new_mesh = {}
        for key, value in mesh.items():
            if torch.is_tensor(value):
                if value.is_floating_point():
                    new_mesh[key] = value.to(target_device, dtype=torch.float32)
                else:
                    new_mesh[key] = value.to(target_device)
            else:
                new_mesh[key] = value
        moved.append(new_mesh)
    return moved


def build_model(example: dict, num_classes: int, args: argparse.Namespace, device: torch.device) -> nn.Module:
    hodgenet_model = HodgeNetModel(
        example["int_edge_features"].shape[1],
        example["triangle_features"].shape[1],
        num_output_features=args.n_out_features,
        mesh_feature=True,
        num_eigenvectors=args.n_eig,
        num_extra_eigenvectors=args.n_extra_eig,
        num_vector_dimensions=args.num_vector_dimensions,
    )

    model = nn.Sequential(
        hodgenet_model,
        nn.Linear(args.n_out_features * args.num_vector_dimensions**2, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Linear(64, 64),
        nn.BatchNorm1d(64),
        nn.LeakyReLU(),
        nn.Linear(64, num_classes),
    ).to(device, dtype=torch.float32)
    return model


def main() -> int:
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

    if not args.input.exists():
        print(json.dumps({"error": f"input not found: {args.input}"}), flush=True)
        return 1

    if not args.model.exists():
        print(json.dumps({"error": f"model checkpoint not found: {args.model}"}), flush=True)
        return 1

    features = ["vertices", "normals"]
    dataset = HodgenetMeshDataset(
        [str(args.input)],
        decimate_range=None,
        edge_features_from_vertex_features=features,
        triangle_features_from_vertex_features=features,
        max_stretch=0,
        random_rotation=False,
        mesh_features={"category": [0]},
        normalize_coords=True,
    )

    example = dataset[0]

    checkpoint = torch.load(args.model, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_classes = None
    for key, tensor in state_dict.items():
        if key.endswith(".weight") and getattr(tensor, "ndim", 0) == 2:
            num_classes = tensor.shape[0]
    if num_classes is None:
        raise RuntimeError("Unable to infer number of classes from checkpoint")

    model = build_model(example, num_classes, args, device)
    model.load_state_dict(state_dict)
    model.eval()

    batch = move_batch_to_device([example], device)
    with torch.no_grad():
        logits = model(batch)[0]
        probs = torch.softmax(logits, dim=0)
        pred_idx = int(torch.argmax(probs).item())

    mapping = load_label_map(args.data_root)
    result = {
        "input": str(args.input),
        "checkpoint": str(args.model),
        "pred_class": pred_idx,
        "label_name": mapping.get(pred_idx) if mapping else None,
        "confidence": float(probs[pred_idx].item()),
        "logits": probs.tolist(),
        "n_classes": num_classes,
        "device": str(device),
    }

    print(json.dumps(result), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
