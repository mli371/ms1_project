import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from subject_api import SubjectRunner
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

WEIGHTS_HASH_CACHE: Dict[str, Optional[str]] = {}

ENTRY_PRIORITY = [
    "eval",
    "evaluate",
    "infer",
    "inference",
    "classification",
    "segmentation",
    "reconstruct",
    "metrics",
    "demo",
]


def canonical_key(value: Optional[str]) -> str:
    if not value:
        return ""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def list_dataset_entries(subject: Dict) -> List[Dict]:
    raw = subject.get("datasets")
    if raw is None:
        raw = subject.get("dataset", [])
    if isinstance(raw, (str, dict)):
        raw = [raw]

    entries: List[Dict] = []
    for item in raw or []:
        if isinstance(item, str):
            entries.append({"name": item})
        elif isinstance(item, dict):
            entry = {k: v for k, v in item.items()}
            if "name" not in entry and "key" in entry:
                entry["name"] = entry["key"]
            entries.append(entry)
    return entries


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
    if not os.path.exists(jsonl_path):
        logging.warning("JSONL not found for aggregation: %s", jsonl_path)
        return
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    agg: Dict[Tuple[str, str], Dict] = {}
    for r in rows:
        key = (r.get("subject"), r.get("tool"))
        g = agg.setdefault(
            key,
            {
                "generated_total": 0,
                "valid_count": 0,
                "invalid_count": 0,
                "crash_count": 0,
                "execution_time_sec_sum": 0.0,
                "records": 0,
            },
        )
        g["generated_total"] += int(r.get("generated_total", 0))
        g["valid_count"] += int(r.get("valid_count", 0))
        g["invalid_count"] += int(r.get("invalid_count", 0))
        g["crash_count"] += int(r.get("crash_count", 0))
        g["execution_time_sec_sum"] += float(r.get("execution_time_sec", 0.0))
        g["records"] += 1

    fieldnames = [
        "subject",
        "tool",
        "generated_total",
        "valid_count",
        "invalid_count",
        "valid_rate",
        "invalid_rate",
        "crash_count",
        "mean_execution_time_sec",
    ]
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
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
            writer.writerow(row)

    with open(md_out, "w") as f:
        f.write("| subject | tool | generated_total | valid_count | invalid_count | valid_rate | invalid_rate | crash_count | mean_execution_time_sec |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for (subject, tool), g in sorted(agg.items()):
            gen_total = g["generated_total"] or 1
            f.write(
                f"| {subject} | {tool} | {g['generated_total']} | {g['valid_count']} | {g['invalid_count']} | "
                f"{g['valid_count']/gen_total:.3f} | {g['invalid_count']/gen_total:.3f} | {g['crash_count']} | "
                f"{(g['execution_time_sec_sum']/max(1, g['records'])):.2f} |\n"
            )


def select_dataset(subject: Dict, seed: Dict) -> Dict:
    entries = list_dataset_entries(subject)
    if not entries:
        fallback = seed.get("dataset") or "unknown"
        return {"name": fallback}

    seed_dataset = seed.get("dataset")
    if seed_dataset:
        seed_key = canonical_key(seed_dataset)
        for entry in entries:
            for candidate in (
                entry.get("name"),
                entry.get("key"),
            ):
                if canonical_key(candidate) == seed_key:
                    return entry
    return entries[0]


def select_entry_template(subject: Dict, dataset_entry: Dict) -> Tuple[str, str]:
    templates = subject.get("entry_template", {}) or {}
    if not isinstance(templates, dict) or not templates:
        raise ValueError(f"Subject {subject.get('name')} missing entry_template")
    preferred = dataset_entry.get("entry") or subject.get("default_entry")
    if preferred and preferred in templates:
        return preferred, templates[preferred]
    for key in ENTRY_PRIORITY:
        if key in templates:
            return key, templates[key]
    key, value = next(iter(templates.items()))
    return key, value


def select_weights(subject: Dict, dataset_entry: Dict) -> Optional[str]:
    dataset_override = dataset_entry.get("weights")
    if dataset_override:
        return dataset_override

    weights = subject.get("weights") or {}
    if not isinstance(weights, dict) or not weights:
        return None
    key_candidates = [
        dataset_entry.get("weights_key"),
        dataset_entry.get("name"),
        dataset_entry.get("key"),
    ]
    for candidate in key_candidates:
        if not candidate:
            continue
        target = canonical_key(candidate)
        for key, path in weights.items():
            if canonical_key(key) == target:
                return path
    return next(iter(weights.values()))


def resolve_dataset_root(dataset_entry: Dict, datasets_map: Dict[str, Any]) -> str:
    override = dataset_entry.get("data_root")
    if override:
        return override
    for candidate in (dataset_entry.get("name"), dataset_entry.get("key")):
        if not candidate:
            continue
        value = datasets_map.get(candidate)
        if isinstance(value, dict):
            path = value.get("path")
        else:
            path = value
        if path:
            return path
    return ""


def normalise_metrics(metrics: Dict) -> Dict:
    defaults = {
        "generated_total": 0,
        "valid_count": 0,
        "invalid_count": 0,
        "crash_count": 0,
        "execution_time_sec": 0.0,
    }
    for key, value in defaults.items():
        metrics.setdefault(key, value)

    if metrics.get("invalid_count") == 0 and metrics.get("generated_total") and metrics.get("valid_count"):
        metrics["invalid_count"] = int(metrics["generated_total"]) - int(metrics["valid_count"])

    breakdown = metrics.get("invalid_breakdown") or {}
    breakdown = {k: int(breakdown.get(k, 0)) for k in ["parse_error", "runtime_error", "timeout", "constraint_fail"]}
    metrics["invalid_breakdown"] = breakdown

    for key in ["t_graph_build", "t_mutate", "t_repair"]:
        if key in metrics:
            try:
                metrics[key] = float(metrics[key])
            except (TypeError, ValueError):
                metrics[key] = 0.0
    return metrics


def sha256_of(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    if not os.path.exists(path):
        return None
    if path in WEIGHTS_HASH_CACHE:
        return WEIGHTS_HASH_CACHE[path]
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        WEIGHTS_HASH_CACHE[path] = digest
        return digest
    except OSError:
        WEIGHTS_HASH_CACHE[path] = None
        return None


def run_tuple(
    subject_spec: Dict,
    seed: Dict,
    policy: Dict,
    tool_key: str,
    datasets_map: Dict[str, Any],
) -> Dict:
    dataset_entry = select_dataset(subject_spec, seed)
    dataset_name = dataset_entry.get("name") or dataset_entry.get("key") or "unknown"
    dataset_root = resolve_dataset_root(dataset_entry, datasets_map)
    entry_name, entry_template = select_entry_template(subject_spec, dataset_entry)
    weights_path = select_weights(subject_spec, dataset_entry)

    subject_extra = subject_spec.get("extra") or {}
    dataset_extra = dataset_entry.get("extra") or {}
    merged_extra = {**subject_extra, **dataset_extra}

    device = dataset_entry.get("device", subject_spec.get("device", ""))

    runner = SubjectRunner(
        spec=subject_spec,
        dataset_root=dataset_root,
        entry_name=entry_name,
        entry_template=entry_template,
        device=device,
        weights_path=weights_path,
        extra=merged_extra,
    )

    tool_fn = GENS[tool_key]
    status = "ok"
    error_message = None
    try:
        metrics, meta = tool_fn(
            subject=runner,
            seed_path=seed["path"],
            time_budget_sec=int(policy["time_budget_sec"]),
            threads=int(policy["threads"]),
            mutation_budget=int(policy["mutation_budget"]),
        )
    except Exception as exc:
        logging.exception(
            "Tool '%s' failed for subject '%s' seed '%s'",
            tool_key,
            subject_spec.get("name"),
            seed.get("id"),
        )
        status = "error"
        error_message = str(exc)
        metrics = {
            "generated_total": 0,
            "valid_count": 0,
            "invalid_count": 0,
            "invalid_breakdown": {
                "parse_error": 0,
                "runtime_error": 0,
                "timeout": 0,
                "constraint_fail": 0,
            },
            "crash_count": 0,
            "execution_time_sec": 0.0,
        }
        meta = {"tool_version": None, "tool_args": None}

    metrics = normalise_metrics(metrics)

    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    record = {
        "ts": now,
        "subject": subject_spec.get("name"),
        "task": subject_spec.get("task"),
        "dataset": dataset_name,
        "tool": tool_key,
        "seed_id": seed.get("id"),
        "seed_category": seed.get("category"),
        **{k: metrics.get(k) for k in [
            "generated_total",
            "valid_count",
            "invalid_count",
            "invalid_breakdown",
            "crash_count",
            "execution_time_sec",
            "t_graph_build",
            "t_mutate",
            "t_repair",
        ] if k in metrics},
        "device": device,
        "threads": int(policy["threads"]),
        "mutation_budget": int(policy["mutation_budget"]),
        "tool_args": meta.get("tool_args"),
        "tool_version": meta.get("tool_version"),
        "subject_commit": subject_spec.get("commit"),
        "subject_repo": subject_spec.get("repo_url"),
        "subject_path": subject_spec.get("path"),
        "entry_name": entry_name,
        "entry_template": entry_template,
        "data_root": dataset_root,
        "weights_path": weights_path,
        "weights_sha256": sha256_of(weights_path),
        "hostname": socket.gethostname(),
        "status": status,
        "eval_mode": dataset_entry.get("eval_mode") or subject_spec.get("eval_mode", "default"),
    }

    if runner.last_command:
        record["last_command"] = runner.last_command
    if runner.last_inputs:
        record["command_inputs"] = runner.last_inputs
    if dataset_entry:
        record["dataset_config"] = {k: v for k, v in dataset_entry.items() if k != "extra"}
    if error_message:
        record["error_message"] = error_message[:512]

    return record


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", required=True)
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument(
        "--tools",
        required=True,
        help="Comma-separated: random,afl,rand_graph,rand_mesh",
    )
    ap.add_argument("--out", required=True, help="JSONL output path")
    ap.add_argument("--aggregate", help="CSV output path for aggregation")
    ap.add_argument("--time-override", type=int, help="Override time_budget_sec for smoke tests")
    ap.add_argument("--resume", action="store_true", help="Skip completed tuples found in JSONL")
    return ap.parse_args()


def main():
    args = parse_args()
    subjects_cfg = load_yaml(args.subjects)
    datasets_cfg = load_yaml(args.datasets)
    seeds_cfg = load_yaml(args.seeds)
    policy = load_yaml(args.policy)

    if args.time_override is not None:
        policy["time_budget_sec"] = int(args.time_override)

    setup_logging(policy.get("log_level", "INFO"))

    subjects: List[Dict] = subjects_cfg.get("subjects", [])
    seeds: List[Dict] = seeds_cfg.get("seeds", [])
    datasets_map: Dict[str, Any] = datasets_cfg.get("datasets", {})

    tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    for tool in tools:
        if tool not in GENS:
            raise SystemExit(f"Unknown tool: {tool}")

    done_keys = set()
    if args.resume and os.path.exists(args.out):
        with open(args.out, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_keys.add((rec.get("subject"), rec.get("tool"), rec.get("seed_id")))
                except Exception:
                    continue

    ensure_parent(args.out)

    tuples: List[Tuple[Dict, Dict, str]] = []
    for subject in subjects:
        for seed in seeds:
            for tool in tools:
                key = (subject.get("name"), tool, seed.get("id"))
                if key in done_keys:
                    logging.info("Resuming: skip completed %s", key)
                    continue
                tuples.append((subject, seed, tool))

    logging.info("Total tuples scheduled: %d", len(tuples))

    with concurrent.futures.ThreadPoolExecutor(max_workers=int(policy.get("threads", 4))) as executor:
        futures = [
            executor.submit(run_tuple, subject, seed, policy, tool, datasets_map)
            for subject, seed, tool in tuples
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                logging.exception("Run failed: %s", exc)
                continue
            write_jsonl(args.out, record)

    if args.aggregate:
        csv_out = args.aggregate
        md_out = os.path.splitext(csv_out)[0] + ".md"
        aggregate_to_csv(args.out, csv_out, md_out)
        logging.info("Aggregation written: %s and %s", csv_out, md_out)


if __name__ == "__main__":
    main()
