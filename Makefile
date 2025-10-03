PY=python

SUBJECTS=configs/subjects.yml
SEEDS=configs/seeds.yml
POLICY=configs/ms1_policy.yml

SMOKE_OUT=logs/raw/smoke/run.jsonl
SMOKE_CSV=logs/agg/smoke_summary.csv

FULL_OUT=logs/raw/full_$(shell date +%F)/run.jsonl
FULL_CSV=logs/agg/full_summary.csv

.PHONY: smoke full aggregate clean

smoke:
	$(PY) scripts/ms1_runner.py \
	  --subjects $(SUBJECTS) \
	  --seeds $(SEEDS) \
	  --policy $(POLICY) \
	  --tools random,afl,rand_graph,rand_mesh \
	  --time-override 300 \
	  --out $(SMOKE_OUT) \
	  --aggregate $(SMOKE_CSV)

full:
	$(PY) scripts/ms1_runner.py \
	  --subjects $(SUBJECTS) \
	  --seeds $(SEEDS) \
	  --policy $(POLICY) \
	  --tools random,afl,rand_graph,rand_mesh \
	  --out $(FULL_OUT) \
	  --aggregate $(FULL_CSV)

aggregate:
	$(PY) scripts/ms1_runner.py --subjects $(SUBJECTS) --seeds $(SEEDS) --policy $(POLICY) --tools random --out $(SMOKE_OUT) --aggregate $(SMOKE_CSV)

clean:
	rm -rf logs/raw/* logs/agg/*

