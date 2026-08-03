#!/usr/bin/env python3
"""Convert raw JSONL event logs into Parquet tables + a summary CSV.

Produces (plan Stage 1 output layout):
    tensor_events.parquet     one row per tensor_ready
    s3_requests.parquet       one row per S3 request (joined start/headers/
                              first_byte/end events)
    metadata_queries.parquet  one row per metadata query
    resources.parquet         resource samples
    summary.csv               per-run medians/percentiles

Usage:
    python eval/scripts/summarize.py --dir eval/results/single_client
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import pandas as pd


def load_events(paths):
    rows = []
    for p in paths:
        with open(p) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # torn tail line from a crashed run
    return pd.DataFrame(rows)


def join_s3_requests(ev: pd.DataFrame) -> pd.DataFrame:
    s3 = ev[ev.event.str.startswith("s3_", na=False)].copy()
    if s3.empty:
        return s3
    out = []
    keycols = ["run_id", "request_id"]
    for (run_id, rid), g in s3.groupby(keycols):
        d = {"run_id": run_id, "request_id": rid}
        for _, r in g.iterrows():
            d.setdefault("key", r.get("key"))
            d.setdefault("operation", r.get("operation"))
            d.setdefault("tensor_name", r.get("tensor_name"))
            d[r["event"] + "_ns"] = r["ts_ns"]
            for f in ("payload_bytes", "http_status", "aws_request_id",
                      "retry_attempts", "cache_hit"):
                if f in r and pd.notna(r.get(f)):
                    d[f] = r[f]
        if "s3_request_start_ns" in d and "s3_headers_ns" in d:
            d["headers_ms"] = (d["s3_headers_ns"]
                               - d["s3_request_start_ns"]) / 1e6
        if "s3_request_start_ns" in d and "s3_request_end_ns" in d:
            d["total_ms"] = (d["s3_request_end_ns"]
                             - d["s3_request_start_ns"]) / 1e6
        out.append(d)
    return pd.DataFrame(out)


def pctiles(s: pd.Series) -> dict:
    if s.empty:
        return {}
    return {
        "p50": s.median(), "p90": s.quantile(0.9), "p95": s.quantile(0.95),
        "p99": s.quantile(0.99), "min": s.min(), "max": s.max(),
        "cv": s.std() / s.mean() if s.mean() else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="eval/results/single_client")
    args = ap.parse_args()

    event_files = sorted(glob.glob(os.path.join(args.dir, "events_*.jsonl")))
    if not event_files:
        print(f"no events_*.jsonl under {args.dir}")
        return 1
    ev = load_events(event_files)

    tensors = ev[ev.event == "tensor_ready"].copy()
    if not tensors.empty:
        tensors.to_parquet(os.path.join(args.dir, "tensor_events.parquet"))
    s3req = join_s3_requests(ev)
    if not s3req.empty:
        s3req.to_parquet(os.path.join(args.dir, "s3_requests.parquet"))
    meta = ev[ev.event.str.startswith("metadata_query", na=False)]
    if not meta.empty:
        meta.to_parquet(os.path.join(args.dir, "metadata_queries.parquet"))
    res = ev[ev.event == "resource_sample"]
    if not res.empty:
        res.to_parquet(os.path.join(args.dir, "resources.parquet"))

    runs_path = os.path.join(args.dir, "model_runs.jsonl")
    if os.path.exists(runs_path):
        runs = pd.read_json(runs_path, lines=True)
        rows = []
        for (system, model), g in runs.groupby(["system", "model_id"]):
            row = {"system": system, "model_id": model,
                   "n_runs": len(g), "n_ok": int(g.ok.sum())}
            row.update({f"e2e_ms_{k}": v for k, v in
                        pctiles(g[g.ok].model_e2e_ms).items()})
            ok = g[g.ok]
            if not ok.empty:
                row["logical_bytes"] = int(ok.logical_model_bytes.iloc[0])
                row["s3_bytes_mean"] = float(ok.s3_payload_bytes.mean())
                row["gets_mean"] = float(ok.s3_get_count.mean())
                if row["logical_bytes"]:
                    row["read_amplification"] = (row["s3_bytes_mean"]
                                                 / row["logical_bytes"])
            rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(args.dir, "summary.csv"), index=False)
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nwrote parquet + summary under {args.dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
