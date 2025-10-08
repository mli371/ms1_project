PY=python

SUBJECTS=configs/subjects.yml
SEEDS=configs/seeds.yml
POLICY=configs/ms1_policy.yml
DATASETS=configs/datasets.yml

SMOKE_OUT=logs/raw/smoke/run.jsonl
SMOKE_CSV=logs/agg/smoke_summary.csv

FULL_OUT=logs/raw/full_$(shell date +%F)/run.jsonl
FULL_CSV=logs/agg/full_summary.csv

.PHONY: smoke full aggregate clean

smoke:
	$(PY) scripts/ms1_runner.py \
	  --subjects $(SUBJECTS) \
	  --datasets $(DATASETS) \
	  --seeds $(SEEDS) \
	  --policy $(POLICY) \
	  --tools random,afl,rand_graph,rand_mesh \
	  --time-override 300 \
	  --out $(SMOKE_OUT) \
	  --aggregate $(SMOKE_CSV)

full:
	$(PY) scripts/ms1_runner.py \
	  --subjects $(SUBJECTS) \
	  --datasets $(DATASETS) \
	  --seeds $(SEEDS) \
	  --policy $(POLICY) \
	  --tools random,afl,rand_graph,rand_mesh \
	  --out $(FULL_OUT) \
	  --aggregate $(FULL_CSV)

aggregate:
	$(PY) scripts/ms1_runner.py --subjects $(SUBJECTS) --datasets $(DATASETS) --seeds $(SEEDS) --policy $(POLICY) --tools random --out $(SMOKE_OUT) --aggregate $(SMOKE_CSV)

clean:
	rm -rf logs/raw/* logs/agg/*

run-p2m:
	MS_ROOT=$$(pwd) python -m ms1.scripts.ms1_runner \
	 --subjects ms1/configs/subjects.yml \
	 --datasets ms1/configs/datasets.yml \
	 --policy   ms1/configs/ms1_policy.yml \
	 --topic point2mesh --max-prompts 1 \
	 --out ms1/logs/ms1_point2mesh_smoke.jsonl

run-topic:
	MS_ROOT=$$(pwd) python -m ms1.scripts.ms1_runner \
	 --subjects ms1/configs/subjects.yml \
	 --datasets ms1/configs/datasets.yml \
	 --policy   ms1/configs/ms1_policy.yml \
	 --topic $${TOPIC} --max-prompts 1 \
	 --out ms1/logs/ms1_$${TOPIC}_smoke.jsonl
