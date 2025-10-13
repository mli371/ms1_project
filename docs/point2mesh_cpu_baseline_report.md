# Point2Mesh (CPU Baseline) – Progress & Findings

## 1. Background
- **Project**: `ms1/` orchestrates baseline evaluations for multiple 3D subjects (MeshCNN, Point2Mesh, MeshSDF, etc.).  
- **Objective**: Establish a reliable CPU baseline for Point2Mesh on COSEG (chair, vase, tele-aliens) using reproducible preprocessing, training logs, and visual artefacts.  
- **Scope**: All work performed in repo root `ms1/`; external dependencies are handled via the `point2mesh` conda environment.

## 2. Repository Snapshot (Point2Mesh-relevant)
| Path | Purpose |
| --- | --- |
| `subjects_src/point2mesh/` | Upstream Point2Mesh checkout. |
| `workdir/COSEG/Point2Mesh/` | Intermediate outputs (normalized PLYs, meshes, reconstructions, visualizations). |
| `configs/subjects.yml` | MS1 entry templates; current CPU baseline points to 160-iteration configs for chair/vase/tele. |
| `ms1/logs/` / `workdir/.../ms1_coseg_*` | Execution logs and per-run artefacts. |
| `scripts/normalize_with_normals.py`, `scripts/icp_align_mesh_to_pcd.py` | Utilities for point-cloud normalization and ICP alignment (matplotlib/Open3D variants retained). |

## 3. Environment State (CPU-only)
```
python: /home/mingyang/miniconda3/envs/point2mesh/bin/python
torch  : 2.3.1+cpu
torchvision / torchaudio : 0.18.1+cpu / 2.3.1+cpu
pytorch3d: 0.7.8 (built from source; CPU backend)
open3d  : 0.19.0
cuda_available: False   (intentionally CPU-only)
```
> Additional packages: `imageio`, `imageio-ffmpeg`, `matplotlib`, `numpy`, `trimesh`, etc., installed as needed for preprocessing/visualization.

## 4. Data Preparation
1. **Normalized point cloud (chair)**  
   - `scripts/normalize_with_normals.py` (Open3D) produces `workdir/COSEG/Point2Mesh/chair_001_norm_withnorm.ply` with unit-sphere scaling and vertex normals.
2. **Mesh alignment**  
   - `scripts/icp_align_mesh_to_pcd.py --n-points 20000 --threshold-scale 1.5` aligns `initmesh.obj` to the normalized PLY.  
   - Final aligned mesh (with recomputed normals): `workdir/COSEG/Point2Mesh/coseg_ply/chairs/initmesh_icp_refined_normals.obj`.

## 5. Baseline Runs (CPU, 160 iterations)
Executed via `subjects_src/point2mesh/main.py` using `/home/mingyang/miniconda3/envs/point2mesh/bin/python`.

| Sample | Input Point Cloud | Init Mesh | Output Dir | min_loss | final_loss | Log |
| --- | --- | --- | --- | --- | --- | --- |
| chair | `chair_001_norm_withnorm.ply` | `initmesh_icp_refined_normals.obj` | `workdir/.../ms1_coseg_chair_cpu_160/` | 0.1058 | 0.1064 | `stdout.log` |
| vase | `coseg_ply/vases/sample.ply` | `coseg_ply/vases/initmesh.obj` | `workdir/.../ms1_coseg_vase_cpu_160/` | –0.1075 | –0.1055 | `stdout.log` |
| tele | `coseg_ply/tele_aliens/sample.ply` | `coseg_ply/tele_aliens/initmesh.obj` | `workdir/.../ms1_coseg_tele_cpu_160/` | 0.0191 | 0.0296 | `stdout.log` |

Notes:
- Loss values stabilise around 0.10 (chair), –0.105 (vase), 0.03 (tele).  
- Chair run leverages normalized PLY + ICP-aligned mesh, yet remains above the aspirational <0.05 threshold (further tuning required).  
- Vase loss negative (expected for this objective/implementation).  
- Tele converges near 0.03 but oscillates slightly in later iterations.

## 6. GPU Attempt (Abandoned)
- Patched `PartMesh` to push tensors onto GPU.  
- Upgraded to `torch 2.3.1+cu121`; attempted to install GPU PyTorch3D.  
- Blocked by lack of prebuilt wheels for Python 3.8; fallback to source build still yielded CPU-only PyTorch (knn CUDA not compiled).  
- Decision: revert to stable CPU toolchain; restore `mesh.py` from backup.

## 7. Visual Artefacts
Generated via matplotlib-based renderer (CPU-friendly):

| Sample | Directory | Assets |
| --- | --- | --- |
| chair | `workdir/.../ms1_coseg_chair_cpu_160/viz/` | `front.png`, `side.png`, `iso.png`, `turntable.gif`, `pcd_mesh_overlay.png` |
| vase | `workdir/.../ms1_coseg_vase_cpu_160/viz/` | `front.png`, `side.png`, `iso.png`, `turntable.gif` |
| tele | `workdir/.../ms1_coseg_tele_cpu_160/viz/` | `front.png`, `side.png`, `iso.png`, `turntable.gif` |

> The chair overlay highlights alignment between the normalized point cloud and the final reconstruction.

## 8. Documentation Updates
- `ms1/docs/ms1_status.md` includes a “Point2Mesh (CPU baseline)” entry summarising losses/logs, and mentions chair’s normalized/ICP setup versus default vase/tele inputs.  
- Baseline results support quick references for future experiments or handoff.

## 9. Outstanding Work / Future Directions
1. **Loss Reduction**  
   - Explore longer CPU runs (>160 iter), alternative loss weightings, or enhanced ICP / normal refinement to drive chair closer to <0.05.  
   - Evaluate additional COSEG samples (beyond single exemplars) for variability analysis.
2. **GPU Path (optional)**  
   - Revisit with Python ≥3.9 to access official PyTorch3D GPU wheels, then retry GPU fine-tuning when hardware time permits.
3. **Integration & Automation**  
   - Incorporate the preprocessing scripts (`normalize_with_normals.py`, `icp_align_mesh_to_pcd.py`) into automated pipelines or MS1 dataset hooks.  
   - Capture hyperparameters / commands in `subjects.yml` comments for reproducibility.
4. **Reporting Enhancements**  
   - Export metrics to CSV/Markdown for aggregated tracking.  
   - Embed generated PNG/GIF in README-style briefs for quick visual QA.

## 10. Quick Reference – Key Paths
- Logs:  
  - Chair: `workdir/COSEG/Point2Mesh/ms1_coseg_chair_cpu_160/stdout.log`  
  - Vase: `workdir/COSEG/Point2Mesh/ms1_coseg_vase_cpu_160/stdout.log`  
  - Tele: `workdir/COSEG/Point2Mesh/ms1_coseg_tele_cpu_160/stdout.log`
- Inputs:  
  - Normalized PLY: `workdir/COSEG/Point2Mesh/chair_001_norm_withnorm.ply`  
  - Aligned Mesh: `workdir/COSEG/Point2Mesh/coseg_ply/chairs/initmesh_icp_refined_normals.obj`
- Visuals: see Section 7.

---
**Snapshot Date**: 2025-10-12  
**Maintained By**: point2mesh CPU baseline effort (MS1 team)
