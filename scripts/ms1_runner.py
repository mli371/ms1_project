import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# Ensure `MSproject/ms1` root is on sys.path so `scripts.*` imports work when invoking as a file
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from subject_api import SubjectRunner

# Generators registry
from generators import random_testing as gen_random
from generators import afl_wrapper as gen_afl
from generators import random_graph as gen_rgraph
from generators import random_mesh as gen_rmesh


GENS = {
    "random": gen_random.run,
    "afl": gen_afl.run,
    "rand_graph": gen_rgraph.run,
    "rand_mesh": gen_rmesh.run,
}


def setup_logging(level: str):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str, obj: Dict):
    ensure_parent(path)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def aggregate_to_csv(jsonl_path: str, csv_out: str, md_out: str):
    ensure_parent(csv_out)
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    # group by (subject, tool)
    agg: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (r.get("subject"), r.get("tool"))
        g = agg.setdefault(key, {
            "generated_total": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "crash_count": 0,
            "execution_time_sec_sum": 0.0,
            "records": 0,
        })
        g["generated_total"] += int(r.get("generated_total", 0))
        g["valid_count"] += int(r.get("valid_count", 0))
        g["invalid_count"] += int(r.get("invalid_count", 0))
        g["crash_count"] += int(r.get("crash_count", 0))
        g["execution_time_sec_sum"] += float(r.get("execution_time_sec", 0.0))
        g["records"] += 1

    fieldnames = [
        "subject", "tool", "generated_total", "valid_count", "invalid_count",
        "valid_rate", "invalid_rate", "crash_count", "mean_execution_time_sec",
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for (subject, tool), g in sorted(agg.items()):
            gen_total = g["generated_total"] or 1
            row = {
                "subject": subject,
                "tool": tool,
                "generated_total": g["generated_total"],
                "valid_count": g["valid_count"],
                "invalid_count": g["invalid_count"],
                "valid_rate": g["valid_count"] / gen_total,
                "invalid_rate": g["invalid_count"] / gen_total,
                "crash_count": g["crash_count"],
                "mean_execution_time_sec": g["execution_time_sec_sum"] / max(1, g["records"]),
            }
            w.writerow(row)

    # Markdown table
    with open(md_out, "w") as f:
        f.write("| subject | tool | generated_total | valid_count | invalid_count | valid_rate | invalid_rate | crash_count | mean_execution_time_sec |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for (subject, tool), g in sorted(agg.items()):
            gen_total = g["generated_total"] or 1
            f.write(
                f"| {subject} | {tool} | {g['generated_total']} | {g['valid_count']} | {g['invalid_count']} | "
                f"{g['valid_count']/gen_total:.3f} | {g['invalid_count']/gen_total:.3f} | {g['crash_count']} | "
                f"{(g['execution_time_sec_sum']/max(1,g['records'])):.2f} |\n"
            )


def run_tuple(
    subject_spec: Dict,
    seed: Dict,
    policy: Dict,
    tool_key: str,
) -> Dict:
    subject = SubjectRunner(
        name=subject_spec["name"],
        entry_cmd=subject_spec["entry"],
        device=subject_spec.get("device", ""),
    )
    seed_path = seed["path"]
    tool_fn = GENS[tool_key]

    metrics, meta = tool_fn(
        subject=subject,
        seed_path=seed_path,
        time_budget_sec=int(policy["time_budget_sec"]),
        threads=int(policy["threads"]),
        mutation_budget=int(policy["mutation_budget"]),
    )

    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "ts": now,
        "subject": subject_spec["name"],
        "tool": tool_key,
        "seed_id": seed["id"],
        "seed_category": seed.get("category"),
        **metrics,
        "device": subject_spec.get("device"),
        "threads": int(policy["threads"]),
        "mutation_budget": int(policy["mutation_budget"]),
        "tool_args": meta.get("tool_args"),
        "subject_commit": subject_spec.get("commit", "unknown"),
        "tool_version": meta.get("tool_version"),
        "hostname": socket.gethostname(),
        "status": "ok",
    }
    return record


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tools", required=True, help="Comma-separated: random,afl,rand_graph,rand_mesh")
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--aggregate", required=False, help="CSV output path for aggregation")
    ap.add_argument("--time-override", type=int, default=None, help="Override time_budget_sec for quick smoke tests")
    ap.add_argument("--resume", action="store_true", help="Skip finished (subject,tool,seed) tuples found in JSONL")
    return ap.parse_args()


def main():
    args = parse_args()
    subjects_cfg = load_yaml(args.subjects)
    seeds_cfg = load_yaml(args.seeds)
    policy = load_yaml(args.policy)

    if args.time_override is not None:
        policy["time_budget_sec"] = int(args.time_override)

    setup_logging(policy.get("log_level", "INFO"))

    subjects: List[Dict] = subjects_cfg.get("subjects", [])
    seeds: List[Dict] = seeds_cfg.get("seeds", [])
    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    for t in tools:
        if t not in GENS:
            raise SystemExit(f"Unknown tool: {t}")

    # Resume support
    done_keys = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done_keys.add((r.get("subject"), r.get("tool"), r.get("seed_id")))
                except Exception:
                    continue

    ensure_parent(args.out)

    # Plan execution
    tuples: List[Tuple[Dict, Dict, str]] = []
    for subj in subjects:
        for seed in seeds:
            for tool in tools:
                key = (subj["name"], tool, seed["id"])
                if key in done_keys:
                    logging.info("Resuming: skipping completed %s", key)
                    continue
                tuples.append((subj, seed, tool))

    logging.info("Total tuples to run: %d", len(tuples))

    # Use a thread pool for outer parallelism, inner loops enforce time budgets
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(policy.get("threads", 4))) as ex:
        futs = [ex.submit(run_tuple, subj, seed, policy, tool) for subj, seed, tool in tuples]
        for fut in concurrent.futures.as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                logging.exception("Run failed: %s", e)
                continue
            write_jsonl(args.out, rec)

    if args.aggregate:
        csv_out = args.aggregate
        md_out = os.path.splitext(csv_out)[0] + ".md"
        aggregate_to_csv(args.out, csv_out, md_out)
        logging.info("Aggregation written: %s and %s", csv_out, md_out)


if __name__ == "__main__":
    main()
