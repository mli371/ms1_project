# HodgeNet – ModelNet40-lite GPU Baseline

## 1. Background
- **Project**: This repository orchestrates multi-subject baselines (Point2Mesh, HodgeNet, MeshCNN, …) under a unified MS1 runner.
- **Goal**: Establish a reproducible GPU training recipe for HodgeNet on the lightweight ModelNet40_lite600 split, including environment lock, instrumentation, and training outputs.
- **Scope**: All commands are executed from the repo root. The upstream HodgeNet code sits in `subjects_src/HodgeNet/` (git submodule) with local patches for CUDA handling and diagnostics.

## 2. Repository Snapshot (HodgeNet focus)
| Path | Purpose |
| --- | --- |
| `subjects_src/HodgeNet/` | HodgeNet upstream sources; patched with GPU-safe eigensolver fallback and detailed logging. |
| `subjects_src/HodgeNet/envs/hodgenet-gpu.lock.yml` | Frozen Conda environment for the GPU baseline (CUDA 12.1 / PyTorch 1.9 stack). |
| `workdir/ModelNet40/HodgeNet_lite/` | Intermediate and final training artefacts (`baseline_debug`, `baseline_10`, `baseline_20`). |
| `logs/hodgenet_*.json` | Runtime telemetry: probe timings, batch profiling, eigenflow snapshots, final baseline summaries. |
| `docs/ms1_status.md` | Status dashboard summarising the 20‑epoch baseline and operational notes. |

## 3. Environment Summary
> Exported to `subjects_src/HodgeNet/envs/hodgenet-gpu.lock.yml`.

- **Python / CUDA**: Python 3.9.6, CUDA toolkit 11.1 (runtime compatible with driver CUDA 12.1).
- **Core libs**: PyTorch 1.9.0, torchvision 0.10.0, SciPy 1.7.0, NumPy 1.20.3, igl 2.2.1.
- **Utilities added for instrumentation**: TensorBoard 2.5.0, matplotlib 3.5.3, tqdm 4.62.0.
- **Hardware**: NVIDIA GeForce RTX 4070 Ti SUPER (16 GB). All runs executed with `num_workers=0`, `pin_memory=True` (WSL-friendly settings).

## 4. Data Preparation
1. **Dataset source** – `data/datasets/ModelNet40_lite600/` (600 training meshes, 5 classes). `labels.txt` regenerated locally to list relative OBJ paths.
2. **Offline preprocessing assumptions** – meshes are decimated offline to ≈500 faces; online decimation / augmentation disabled (`decimate_range=None`, `max_stretch=0`, `random_rotation=False`).
3. **Baseline config** – validation performed on remaining meshes per class (shuffle + split handled by the training script).

## 5. Model & Data Artefacts
- **Checkpoints**: Baseline 20e exports `workdir/ModelNet40/HodgeNet_lite/baseline_20/best.pth` (highest validation accuracy) and `.../last.pth` (final epoch), plus `4/9/14/19.pth` for intermediate inspection.
- **Dataset**: Training/validation pulls from `data/datasets/ModelNet40_lite600/`; retain this directory unchanged for MS2 fuzzing / perturbation studies.
- **Inference entry point**: `subjects_src/HodgeNet/train_classification.py` builds the `HodgeNetModel` (see `subjects_src/HodgeNet/hodgenet.py`). For downstream inference, load `best.pth` via `torch.load(...)["model_state_dict"]` and call `HodgeNetModel` directly (batched forward identical to training loop).

## 6. Instrumentation & Logging
- **Eigensolver fallback** – `subjects_src/HodgeNet/hodgeautograd.py` catches ARPACK singular-factor errors, substitutes zero tensors, and records device/dtype metadata before copying to CUDA (see lines 64‑104 and 158‑210 in the patched file).
- **Probe timings** – `subjects_src/HodgeNet/train_classification.py` logs probe A/B/C timestamps, first-batch device distribution, per-batch iteration durations, and aggregates to `logs/hodgenet_probe_timing.json` / `logs/hodgenet_perf_pinmem.json`.
  - `hodgenet_probe_timing.json`: `timestamps` (raw epoch markers) and `durations` (A→B, B→C, etc.).
  - `hodgenet_perf_pinmem.json`: `iteration_log` (per-batch timing records) and phase-level statistics (`stats.train/validation`).
- **Eigenflow report** – `logs/hodgenet_eigenflow_report.json` tracks CPU shared tensors vs CUDA copies, ensuring no CUDA handles are shared across processes post-spawn.
- **Baseline summaries** – `logs/hodgenet_baseline10_result.json`, `logs/hodgenet_baseline20_result.json` capture wall-clock, batch size, accuracy/loss, and environment info.

## 7. Baseline Runs
| Run | Epochs | Batch size | Duration | Best val acc | Final val loss | Output Dir | Logs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Smoke (repeatability) | 1 × 3 | 4 | ~54 s each (σ ≈ 0.36 s) | – | – | `workdir/ModelNet40/HodgeNet_lite/baseline_debug/` | `logs/hodgenet_repeatability.json`, `logs/hodgenet_perf_pinmem.json` |
| Baseline 10e | 10 | 16 | 404 s | 0.25 | 1.61 | `workdir/ModelNet40/HodgeNet_lite/baseline_10/` | `logs/hodgenet_baseline10_result.json` |
| Baseline 20e | 20 | 16 | 802 s | 0.25 | 1.83 | `workdir/ModelNet40/HodgeNet_lite/baseline_20/` | `logs/hodgenet_baseline20_result.json` |

> The 20‑epoch run extends training to yield fuller curves (accuracy plateau ≈0.25). Validation loss drifts upward after epoch 15, indicating overfitting or insufficient regularisation.

### Artefacts (Baseline 20)
- Curves: `workdir/ModelNet40/HodgeNet_lite/baseline_20/acc_loss_curve.png`, `.../train_val_curve.png`
- Checkpoints: `.../best.pth`, `.../last.pth`, plus intermediate `4/9/14/19.pth`
- Runtime log: `.../stdout.log`
- Telemetry: `logs/hodgenet_probe_timing.json`, `logs/hodgenet_perf_pinmem.json`, `logs/hodgenet_eigenflow_report.json`

## 8. Observations
1. **Stability** – No CUDA handle errors after enforcing spawn start method and CPU shared buffers. ARPACK warnings remain but are safely handled by the fallback.
2. **Performance** – With `pin_memory=True`, first training batch ≈3.2 s, subsequent batches ≈2.2 s; validation batches ≈8.5 s due to eigensolver overhead.
3. **Accuracy plateau** – Validation accuracy stagnates near 25%. Loss curves suggest underfitting/overfitting crossover; additional regularisation or learning-rate scheduling may be required.
4. **Repeatability** – Three sequential 1‑epoch runs show <1% wall-clock variance and consistent probe timings (see `logs/hodgenet_repeatability.json`).

## 9. Next Steps
1. **Hyper-parameter sweeps** – Explore learning-rate decay, weight decay, or increased eigenvector counts to push accuracy beyond 0.25.
2. **Validation cadence** – Increase `val_every` for longer runs to monitor drift and catch potential degeneracy earlier.
3. **ARPACK investigations** – Analyse singular-matrix cases (recorded in eigenflow report) to identify problematic meshes; consider preconditioning or alternative eigensolvers.
4. **Runner integration** – Wire HodgeNet baselines into the MS1 runner once dataset preprocessing is automated for the full ModelNet40 split.

## 10. Quick Reference
- **Command (20e)**:
  ```bash
  PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
  conda run -n HodgeNet \
    python subjects_src/HodgeNet/train_classification.py \
      --data "$PWD/data/datasets/ModelNet40_lite600" \
      --out "$PWD/workdir/ModelNet40/HodgeNet_lite/baseline_20" \
      --n_epochs 20 --bs 16 --num_workers 0 --pin_memory
  ```
- **Environment lock**: `subjects_src/HodgeNet/envs/hodgenet-gpu.lock.yml`
- **Primary logs**: `logs/hodgenet_baseline20_result.json`, `logs/hodgenet_perf_pinmem.json`, `logs/hodgenet_eigenflow_report.json`
- **Code references**: `subjects_src/HodgeNet/hodgeautograd.py` (eigensolver fallback), `subjects_src/HodgeNet/train_classification.py` (probes/iteration profiling)

---
**Snapshot Date**: 2025-10-21  
**Maintained By**: MS1 HodgeNet GPU baseline effort
