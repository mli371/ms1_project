# Point2Mesh File & Directory Map

This note lists the Point2Mesh-related assets in `ms1/`, focused on the pieces exercised by the COSEG baselines, preprocessing utilities, and runner wiring.

## 1. Source Tree (`subjects_src/point2mesh/`)
| Path | Role / Notes |
| --- | --- |
| `main.py` | Entry point used by the MS1 runner; now GPU-aware via `--gpu` flag and updated `PartMesh` flow.
| `options.py` | CLI arguments (save paths, reconstruction hyperparameters, GPU flag). Writes `opt.txt` in each run directory.
| `models/` | Neural network and geometry kernels. Notable files used/modified:<br>• `layers/mesh.py` – mesh partitioning/export logic (patched for device-consistent submesh creation).<br>• `losses.py` – Chamfer + beam gap loss used during reconstruction.<br>• `networks/` – convolutional network definitions invoked by `init_net`.
| `utils.py` | Helpers for reading point clouds, manifold upsampling, etc.
| `scripts/` | Upstream example runners and data prep scripts (e.g., `examples/giraffe.sh` leveraged for smoke tests).
| `envs/` | Template/lock files for point2mesh environments (`point2mesh-cpu.lock.yml`, `point2mesh-gpu.lock.yml`).
| `environment.yml` | Original upstream conda recipe (CPU-oriented); superseded by lockfiles.
| `docs/` | Upstream paper assets; not directly touched but kept for reference.

## 2. MS1 Integration Assets
| Path | Description |
| --- | --- |
| `configs/subjects.yml` | Point2Mesh COSEG entries point at our normalized PLY/mesh assets and now call `python main.py --gpu 0 ...`.
| `docs/point2mesh_report.md` | Consolidated CPU + GPU baseline report (environments, commands, metrics, next steps).
| `docs/ms1_status.md` | Status table summarises current Point2Mesh baselines and TODOs.
| `docs/point2mesh_file_structure.md` | *(this file)* quick reference for collaborators.

## 3. Local Tooling (`scripts/`)
| Script | Purpose |
| --- | --- |
| `normalize_with_normals.py` | Open3D-based unit-sphere scaling + normal estimation for COSEG chairs.
| `normalize_ply.py` | Simpler normalization helper retained for experiments.
| `icp_align_mesh_to_pcd.py` | Wrapper to align template meshes to point clouds (uses Open3D ICP).
| `run_chair_debug.sh` | Convenience script for local debug runs.

## 4. Data & Outputs (`workdir/COSEG/Point2Mesh/`)
| Path | Contents |
| --- | --- |
| `chair_001_norm_withnorm.ply`, `chair_001_norm.ply` | Normalised chair point clouds (with / without normals).
| `coseg_ply/` | Staged COSEG samples (`chairs/`, `vases/`, `tele_aliens/` – each containing `sample.ply`, `initmesh*.obj`).
| `ms1_coseg_*_cpu_160/` | CPU baseline outputs (`opt.txt`, `stdout.log`, periodic `recon_iter_XX.obj`, final `last_recon.obj`).
| `ms1_coseg_*_gpu_160/` | GPU baseline outputs mirroring CPU structure, produced with `point2mesh-gpu` env.
| `ms1_smoke_p2m_{cpu,gpu}/` | 10-iteration smoke tests on the giraffe sample.
| `ms1_coseg_*_cpu_gpu_compare.png` | Loss curves comparing CPU vs GPU runs per shape.
| `gpu_cpu_summary.md` | Markdown table summarising min/final losses and log locations.
| Additional folders (`ms1_coseg_chair_*_debug`, `tmp_manual/`) | Intermediate experiments retained for provenance; safe to ignore for baseline reruns.

## 5. Logs (`ms1/logs/`)
| Log | Description |
| --- | --- |
| `point2mesh_smoke.log` | Output from the 10-iteration giraffe smoke test.
| `ms1_point2mesh_smoke.jsonl`, `ms1_point2mesh_coseg_finetune.jsonl` | Runner-emitted telemetry (per iteration summaries across smoke/baseline jobs).

## 6. Environment Specs
| Path | Notes |
| --- | --- |
| `subjects_src/point2mesh/envs/point2mesh-cpu.lock.yml` | CPU conda lockfile (torch 2.3.0+cpu, pytorch3d 0.7.6 cpu wheel).
| `subjects_src/point2mesh/envs/point2mesh-gpu.lock.yml` | GPU lockfile exported after building PyTorch3D 0.7.6 against CUDA 12.1.

## 7. Quick Navigation Commands
```bash
# Inspect recent GPU baseline losses
sed -n '150,161p' workdir/COSEG/Point2Mesh/ms1_coseg_chair_gpu_160/stdout.log

# Compare loss curves visually
xdg-open workdir/COSEG/Point2Mesh/ms1_coseg_chair_cpu_gpu_compare.png

# Re-run alignment utility
python scripts/icp_align_mesh_to_pcd.py \
  workdir/COSEG/Point2Mesh/coseg_ply/chairs/initmesh.obj \
  workdir/COSEG/Point2Mesh/chair_001_norm_withnorm.ply \
  workdir/COSEG/Point2Mesh/coseg_ply/chairs/initmesh_icp_refined_normals.obj
```

