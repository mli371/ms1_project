#!/usr/bin/env bash
set -e
mkdir -p workdir/COSEG/Point2Mesh
cp entry_template.coseg_demo entry_template.coseg_demo.bak || true
python3 scripts/normalize_ply.py subjects_src/point2mesh/data/coseg_demo/chair_001.ply workdir/COSEG/Point2Mesh/chair_001_norm.ply
MS_ROOT=$(pwd)
OUT=logs/chair_debug.$(date +%Y%m%d_%H%M).jsonl
PYTHONPATH=${MS_ROOT} python -m scripts.ms1_runner --topic point2mesh --max-prompts 1 --out ${OUT}
echo "Log written to ${OUT}"
python3 - <<'PY'
# 简短解析输出最后几行（复用 A 的逻辑简化版）
import json
fn = "'''${OUT}'''".strip("'")
try:
    with open(fn) as f:
        lines = f.readlines()[-20:]
        for l in lines:
            print(l.strip())
except Exception as e:
    print("parse error", e)
PY
