MS1 — Run 8 Subjects with Baseline Generators

## Overview
- Orchestrates subject × tool × seed runs with strict budgets via `ms1.scripts.ms1_runner`.
- Currently prioritises Point2Mesh COSEG baselines (CPU + GPU) and HodgeNet ModelNet40-lite GPU runs while keeping other subjects smoke-test ready.
- Emits JSONL per-run records plus aggregated CSV/Markdown for downstream analysis.

## Quick Start
1. **Prepare configs**  
   - Update `configs/subjects.yml`, `configs/datasets.yml`, and (if needed) `configs/seeds.yml`.  
   - Place dataset links under `data/datasets/` and seed files in `data/seeds/`.
2. **Set environments**  
   - Point2Mesh CPU: `conda env create -f subjects_src/point2mesh/envs/point2mesh-cpu.lock.yml`.  
   - Point2Mesh GPU: `conda env create -f subjects_src/point2mesh/envs/point2mesh-gpu.lock.yml` (requires CUDA 12.1-capable driver).  
   - HodgeNet GPU: `conda env create -f subjects_src/HodgeNet/envs/hodgenet-gpu.lock.yml` (PyTorch 1.9 + CUDA 11.1 toolchain).
3. **Launch a smoke run** (from repo root):
   ```bash
   MS_ROOT=$(pwd) python -m ms1.scripts.ms1_runner \
     --subjects ms1/configs/subjects.yml \
     --datasets ms1/configs/datasets.yml \
     --policy   ms1/configs/ms1_policy.yml \
     --topic point2mesh --max-prompts 1 \
     --out ms1/logs/ms1_point2mesh_smoke.jsonl
   ```
   *(Alternatives: `python ms1/scripts/ms1_runner.py ...` or `python tools/run_ms1.py --topic point2mesh --max-prompts 1`.)*

## Documentation & Artifacts
- `docs/point2mesh_report.md` – detailed Point2Mesh CPU/GPU baseline report.  
- `docs/hodgenet_report.md` – HodgeNet ModelNet40-lite GPU baseline report.  
- `docs/point2mesh_file_structure.md` – reference map for Point2Mesh scripts, workdir outputs, and logs.  
- `docs/ms1_status.md` – project-wide status dashboard with latest loss numbers (Point2Mesh + HodgeNet).  
- Comparison plots live in `workdir/COSEG/Point2Mesh/ms1_coseg_*_cpu_gpu_compare.png`; summary table at `workdir/COSEG/Point2Mesh/gpu_cpu_summary.md`.

## Tooling Highlights
- **Preprocessing**:  
  - `scripts/normalize_with_normals.py` – Open3D normalization + normal estimation.  
  - `scripts/icp_align_mesh_to_pcd.py` – ICP alignment of initial meshes to point clouds (used for COSEG chair baseline).
- **Runner configuration**: `configs/subjects.yml` Point2Mesh entries invoke `python main.py --gpu 0 ...`; HodgeNet configs invoke `subjects_src/HodgeNet/train_classification.py` (auto-selects CUDA when available).

## Notes
- Subject entry templates accept placeholders (`{mesh}`, `{weights}`, `{data_root}`, `{device}`); dataset blocks can override defaults.  
- AFL wrapper still simulates if `afl-fuzz` is missing.  
- GPU resources are optional for most subjects, but Point2Mesh COSEG and HodgeNet baselines now leverage CUDA for faster convergence.  
- WSL hosts: keep `num_workers=0` for HodgeNet eigencalc stability; `pin_memory=True` offers measurable throughput gains.
