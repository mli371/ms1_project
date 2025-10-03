MS1 — Run 8 Subjects with Baseline Generators

Overview
- Orchestrates (subject × tool × seed) runs with strict budgets.
- Baseline tools: random, afl, rand_graph, rand_mesh.
- Emits JSONL per-run records and aggregated CSV/Markdown.

Quick Start (smoke)
- Prepare configs in `configs/` and seeds under `data/seeds/`.
- Run:
  `python scripts/ms1_runner.py --subjects configs/subjects.yml --seeds configs/seeds.yml --policy configs/ms1_policy.yml --tools random,afl,rand_graph,rand_mesh --time-override 300 --out logs/raw/smoke/run.jsonl --aggregate logs/agg/smoke_summary.csv`

Notes
- Subject entry commands are executed as provided in `subjects.yml`. The input mesh path is appended to the command line.
- AFL wrapper will simulate if `afl-fuzz` is not available.
- Code avoids GPU requirements; devices are passed through but not enforced.

