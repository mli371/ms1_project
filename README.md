MS1 — Run 8 Subjects with Baseline Generators

Overview
- Orchestrates (subject × tool × seed) runs with strict budgets.
- Baseline tools: random, afl, rand_graph, rand_mesh.
- Emits JSONL per-run records and aggregated CSV/Markdown.

Quick Start (smoke)
- Prepare configs in `configs/` (fill `subjects.yml`, `datasets.yml`, `seeds.yml`) and seeds under `data/seeds/`.
- Run:
  `python scripts/ms1_runner.py --subjects configs/subjects.yml --datasets configs/datasets.yml --seeds configs/seeds.yml --policy configs/ms1_policy.yml --tools random,afl,rand_graph,rand_mesh --time-override 300 --out logs/raw/smoke/run.jsonl --aggregate logs/agg/smoke_summary.csv`

Notes
- Subject entry templates in `subjects.yml` accept placeholders such as `{mesh}`, `{weights}`, `{data_root}`, `{device}`; dataset-specific options (e.g., `entry`, `data_root`, `weights_key`) can be provided under each `datasets` item and override the defaults.
- AFL wrapper will simulate if `afl-fuzz` is not available.
- Code avoids GPU requirements; devices are passed through but not enforced.
