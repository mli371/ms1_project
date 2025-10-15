# Point2Mesh Baselines – CPU & GPU

## 1. Background
- **Project**: `ms1/` orchestrates baseline evaluations for multiple 3D subjects (MeshCNN, Point2Mesh, MeshSDF, etc.).
- **Objective**: Establish reproducible Point2Mesh baselines on COSEG (chair, vase, tele-aliens) across CPU and GPU toolchains, capturing preprocessing, training logs, and visuals.
- **Scope**: All work is performed in repo root `ms1/`; preprocessing utilities and runner configs live alongside the subject code.

## 2. Repository Snapshot (Point2Mesh focus)
| Path | Purpose |
| --- | --- |
| `subjects_src/point2mesh/` | Upstream Point2Mesh checkout (SIGGRAPH 2020).
| `workdir/COSEG/Point2Mesh/` | Intermediate outputs (normalized PLYs, recon meshes, visualisations).
| `configs/subjects.yml` | MS1 runner templates; now wired for `--gpu 0` on COSEG entries.
| `scripts/normalize_with_normals.py`, `scripts/icp_align_mesh_to_pcd.py` | Utilities for point-cloud normalization and ICP alignment (Open3D pipeline).
| `docs/ms1_status.md` | Project-wide status; updated with CPU vs GPU Point2Mesh metrics.

## 3. Environment States
### CPU baseline (`point2mesh`)
```
python: /home/mingyang/miniconda3/envs/point2mesh/bin/python
torch  : 2.3.1+cpu
torchvision / torchaudio : 0.18.1+cpu / 2.3.1+cpu
pytorch3d: 0.7.8 (CPU only)
open3d  : 0.19.0
cuda_available: False
```

### GPU baseline (`point2mesh-gpu`)
```
python: /home/mingyang/miniconda3/envs/point2mesh-gpu/bin/python
torch  : 2.3.1 (cu121)
torchvision / torchaudio : 0.18.1 (cu121) / 2.3.1 (cu121)
pytorch3d: 0.7.6 (built from source with CUDA 12.1)
open3d  : 0.19.0
cuda_available: True  (GeForce RTX 4070 Ti SUPER)
```
> Build notes: PyTorch3D 0.7.6 compiled from source after enabling CUDA toolkit headers (`targets/x86_64-linux/include/cccl`) and pointing `CUB_HOME` to the bundled CCCL cub sources. Environment spec exported to `subjects_src/point2mesh/envs/point2mesh-gpu.lock.yml`.

## 4. Data Preparation Workflow
1. **Normalize point cloud (chair)** – `scripts/normalize_with_normals.py` generates `workdir/COSEG/Point2Mesh/chair_001_norm_withnorm.ply` with unit-sphere scaling + vertex normals.
2. **ICP-align initial mesh** – `scripts/icp_align_mesh_to_pcd.py --n-points 20000 --threshold-scale 1.5` produces `workdir/COSEG/Point2Mesh/coseg_ply/chairs/initmesh_icp_refined_normals.obj` aligned to the normalized chair cloud.
3. **Shared assets** – Vase/tele runs reuse repository-provided point clouds and meshes under `workdir/COSEG/Point2Mesh/coseg_ply/...`.

## 5. Baselines – CPU (160 iterations)
Executed with `python main.py` inside the `point2mesh` environment (`CUDA_VISIBLE_DEVICES=""`).

| Sample | Input Point Cloud | Init Mesh | Output Dir | min_loss | final_loss | Log |
| --- | --- | --- | --- | --- | --- | --- |
| chair | `chair_001_norm_withnorm.ply` | `initmesh_icp_refined_normals.obj` | `workdir/.../ms1_coseg_chair_cpu_160/` | 0.1058 | 0.1064 | `stdout.log` |
| vase | `coseg_ply/vases/sample.ply` | `coseg_ply/vases/initmesh.obj` | `workdir/.../ms1_coseg_vase_cpu_160/` | −0.1075 | −0.1055 | `stdout.log` |
| tele | `coseg_ply/tele_aliens/sample.ply` | `coseg_ply/tele_aliens/initmesh.obj` | `workdir/.../ms1_coseg_tele_cpu_160/` | 0.0191 | 0.0296 | `stdout.log` |

- Loss values plateau around 0.10 (chair), −0.105 (vase), 0.03 (tele).
- Chair benefits from normalization + ICP but still exceeds the <0.05 target at 160 iters.
- Runtime ≈520 s per shape on CPU; checkpoints exported every 10 iterations.

## 6. Baselines – GPU (160 iterations)
Runs use identical commands with `--gpu 0` inside `point2mesh-gpu` (CUDA 12.1).

| Sample | Input Point Cloud | Init Mesh | Output Dir | min_loss | final_loss | Log |
| --- | --- | --- | --- | --- | --- | --- |
| chair | `chair_001_norm_withnorm.ply` | `initmesh_icp_refined_normals.obj` | `workdir/.../ms1_coseg_chair_gpu_160/` | 0.0178 | 0.0222 | `stdout.log` |
| vase | `coseg_ply/vases/sample.ply` | `coseg_ply/vases/initmesh.obj` | `workdir/.../ms1_coseg_vase_gpu_160/` | −0.1570 | −0.1551 | `stdout.log` |
| tele | `coseg_ply/tele_aliens/sample.ply` | `coseg_ply/tele_aliens/initmesh.obj` | `workdir/.../ms1_coseg_tele_gpu_160/` | −0.0185 | 0.0003 | `stdout.log` |

- Chair loss hits 0.0178 (≈6× reduction vs CPU) within the same 160 iterations.
- Vase improves by ~0.05 absolute; tele converges near zero with minor late-iteration drift.
- Runtime ≈45 s per shape; logs are captured per run, with recon exports every 10 iterations.

## 7. CPU ↔ GPU Comparison & Visuals
- Comparison plots and markdown summary: `workdir/COSEG/Point2Mesh/ms1_coseg_*_cpu_gpu_compare.png` and `workdir/COSEG/Point2Mesh/gpu_cpu_summary.md`.
- GPU runs reuse CPU-generated visual pipelines. Additional GPU renders can be produced by re-running the matplotlib scripts under `point2mesh-gpu` (Open3D is GPU-safe for current usage).

## 8. Documentation & Runner Updates
- `configs/subjects.yml` COSEG templates now call `python main.py --gpu 0 ...` so the MS1 runner executes on CUDA by default.
- `docs/ms1_status.md` reflects the new GPU baselines (min/final losses, runtime expectations) and tracks future expansion tasks.

## 9. Outstanding Work / Next Steps
1. **Broader dataset coverage** – Sweep additional COSEG chairs/vases/tele aliens to gauge variance; consider >160 iterations for chair to approach <0.02.
2. **Reporting** – Integrate `gpu_cpu_summary.md` and plots into higher-level dashboards (README, MS1 status) and add inline links to logs for quick checks.
3. **Automation** – Pipe normalization + ICP scripts into the MS1 preprocessing hooks so fresh samples auto-generate aligned meshes before runner execution.
4. **Numerical experiments** – Examine loss-weight tweaks (e.g., normals weighting, local non-uniform term) to stabilise tele’s late-iteration oscillation.

## 10. Quick Reference – Key Paths
- **CPU logs**: `workdir/COSEG/Point2Mesh/ms1_coseg_{chair,vase,tele}_cpu_160/stdout.log`
- **GPU logs**: `workdir/COSEG/Point2Mesh/ms1_coseg_{chair,vase,tele}_gpu_160/stdout.log`
- **Comparison assets**: `workdir/COSEG/Point2Mesh/gpu_cpu_summary.md`, `workdir/COSEG/Point2Mesh/ms1_coseg_*_cpu_gpu_compare.png`
- **Inputs**: `workdir/COSEG/Point2Mesh/chair_001_norm_withnorm.ply`, `workdir/COSEG/Point2Mesh/coseg_ply/*/{sample.ply,initmesh*.obj}`

---
**Snapshot Date**: 2025-10-14  
**Maintained By**: MS1 Point2Mesh baseline effort
