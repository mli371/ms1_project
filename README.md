MS1 — Run 8 Subjects with Baseline Generators

Overview
- Orchestrates (subject × tool × seed) runs with strict budgets.
- Baseline tools: random, afl, rand_graph, rand_mesh.
- Emits JSONL per-run records and aggregated CSV/Markdown.

Quick Start (smoke)
- Prepare configs in `configs/` (fill `subjects.yml`, `datasets.yml`, `seeds.yml`) and seeds under `data/seeds/`.
- Run (from repo root):
  ```bash
  MS_ROOT=$(pwd) python -m ms1.scripts.ms1_runner \
    --subjects ms1/configs/subjects.yml \
    --datasets ms1/configs/datasets.yml \
    --policy   ms1/configs/ms1_policy.yml \
    --topic point2mesh --max-prompts 1 \
    --out ms1/logs/ms1_point2mesh_smoke.jsonl
  ```
  *(If you insist on `python ms1/scripts/ms1_runner.py ...`, the runner will inject the repo root into `PYTHONPATH` and emit a one-time warning. Alternatively, use `python tools/run_ms1.py --topic point2mesh --max-prompts 1`.)*

Notes
- Subject entry templates in `subjects.yml` accept placeholders such as `{mesh}`, `{weights}`, `{data_root}`, `{device}`; dataset-specific options (e.g., `entry`, `data_root`, `weights_key`) can be provided under each `datasets` item and override the defaults.
- AFL wrapper will simulate if `afl-fuzz` is not available.
- Code avoids GPU requirements; devices are passed through but not enforced.
