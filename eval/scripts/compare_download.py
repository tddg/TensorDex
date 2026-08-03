#!/usr/bin/env python3
"""Compare compressed (tensordex_c) vs uncompressed (tensordex) downloads.

Reads model_runs.jsonl + per-run event traces and produces one row per
run with a phase breakdown, then per-(model, system) medians and a
markdown comparison report.

Phase semantics (phases overlap by design — the pipeline decodes/writes
while later fetches are still in flight):
    resolve_ms      SQLite metadata resolution (closure query)
    fetch_span_ms   first s3_request_start -> last s3_request_end
    decode_cpu_ms   summed wall time inside decompress_tensorx_rust
    tail_ms         e2e - (resolve + fetch span): decode/write work that
                    did NOT overlap with network transfer
    net_MBps        s3_payload_bytes / fetch_span
    goodput_MBps    logical_model_bytes / e2e

Usage:
    python eval/scripts/compare_download.py --dir eval/results/single_client \
        --out eval/results/DOWNLOAD_COMPARISON.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

FAMILY = {"llama-3.2-3b": "llama-3.2-3B", "qwen": "Qwen2.5-7B",
          "gemma": "gemma-2-9B", "woona": "gemma-2-9B"}


def family_of(model_id: str) -> str:
    m = model_id.lower()
    for k, v in FAMILY.items():
        if k in m:
            return v
    return "other"


def run_metrics(row: dict, events_dir: str | None = None) -> dict:
    path = row["events"]
    if not os.path.exists(path) and events_dir:
        path = os.path.join(events_dir, os.path.basename(path))
    ev = []
    with open(path) as fh:
        for line in fh:
            try:
                ev.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    df = pd.DataFrame(ev)
    out = {
        "run_id": row["run_id"], "rep": row["rep"],
        "system": row["system"], "model_id": row["model_id"],
        "family": family_of(row["model_id"]),
        "e2e_s": row["model_e2e_ms"] / 1e3,
        "logical_GB": row["logical_model_bytes"] / 1e9,
        "s3_GB": row["s3_payload_bytes"] / 1e9,
        "gets": row["s3_get_count"],
    }
    out["read_amp"] = (row["s3_payload_bytes"] / row["logical_model_bytes"]
                       if row["logical_model_bytes"] else None)

    mq = df[df.event.str.startswith("metadata_query", na=False)]
    if len(mq) >= 2:
        out["resolve_ms"] = (mq[mq.event == "metadata_query_end"].ts_ns.max()
                             - mq[mq.event == "metadata_query_start"]
                             .ts_ns.min()) / 1e6

    starts = df[df.event == "s3_request_start"]
    ends = df[df.event == "s3_request_end"]
    if not starts.empty and not ends.empty:
        span_ns = ends.ts_ns.max() - starts.ts_ns.min()
        out["fetch_span_s"] = span_ns / 1e9
        out["net_MBps"] = row["s3_payload_bytes"] / 1e6 / (span_ns / 1e9)
        # First-byte latency per request.
        hdr = df[df.event == "s3_headers"][["request_id", "ts_ns"]]
        fb = starts[["request_id", "ts_ns"]].merge(
            hdr, on="request_id", suffixes=("_s", "_h"))
        if not fb.empty:
            lat = (fb.ts_ns_h - fb.ts_ns_s) / 1e6
            out["ttfb_ms_p50"] = lat.median()
            out["ttfb_ms_p95"] = lat.quantile(0.95)

    asm = df[df.event == "model_assembly_end"]
    if not asm.empty:
        a = asm.iloc[-1]
        for f in ("decode_cpu_ms", "bytes_written", "shards"):
            if f in a and pd.notna(a.get(f)):
                out[f] = a[f]

    rs = df[df.event == "resource_sample"]
    if len(rs) >= 2:
        dt = (rs.ts_ns.max() - rs.ts_ns.min()) / 1e9
        out["rss_peak_GB"] = rs.rss_bytes.max() / 1e9
        out["proc_cpu_pct_mean"] = rs.proc_cpu_pct.mean()
        if "disk_write_bytes" in rs and rs.disk_write_bytes.notna().any():
            w = rs.disk_write_bytes.dropna()
            out["disk_write_MBps"] = (w.max() - w.min()) / 1e6 / dt
    if "fetch_span_s" in out and "resolve_ms" in out:
        out["tail_s"] = max(0.0, out["e2e_s"] - out["fetch_span_s"]
                            - out["resolve_ms"] / 1e3)
    out["goodput_MBps"] = out["logical_GB"] * 1e3 / out["e2e_s"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", nargs="+",
                    default=["eval/results/single_client"])
    ap.add_argument("--out", default="eval/results/DOWNLOAD_COMPARISON.md")
    args = ap.parse_args()

    frames = []
    for d in args.dir:
        runs_path = os.path.join(d, "model_runs.jsonl")
        if not os.path.exists(runs_path):
            continue
        runs = pd.read_json(runs_path, lines=True)
        runs["_dir"] = d
        frames.append(runs)
    runs = pd.concat(frames, ignore_index=True)
    ok = runs[runs.ok]
    if ok.empty:
        print("no successful runs")
        return 1
    per_run = pd.DataFrame([run_metrics(r, r["_dir"])
                            for _, r in ok.iterrows()])
    # The server path's rep 0 is cold-cache, rep 1 warm-cache — split
    # them into distinct systems so medians never blend the two regimes.
    srv = per_run.system == "tensordex_srv"
    per_run.loc[srv & (per_run.rep == 0), "system"] = "tensordex_srv_cold"
    per_run.loc[srv & (per_run.rep > 0), "system"] = "tensordex_srv_warm"
    per_run.to_csv(os.path.join(args.dir[0], "per_run_metrics.csv"),
                   index=False)

    med = per_run.groupby(["family", "model_id", "system"]).median(
        numeric_only=True).reset_index()
    med.to_csv(os.path.join(args.dir[0], "per_model_medians.csv"),
               index=False)

    # Pivot: compressed vs uncompressed side by side.
    lines = ["# Compressed vs uncompressed download (single client)\n"]
    n_reps = per_run.groupby(["model_id", "system"]).size().min()
    lines.append(f"Medians over reps (min reps per cell: {n_reps}). "
                 "tensordex = uncompressed dedup store (`tensordb/`), "
                 "tensordex_c = delta-compressed hub "
                 "(`compressed_eval/`).\n")
    cols = ["e2e_s", "logical_GB", "s3_GB", "gets", "read_amp",
            "resolve_ms", "fetch_span_s", "net_MBps", "decode_cpu_ms",
            "tail_s", "ttfb_ms_p50", "goodput_MBps", "rss_peak_GB",
            "proc_cpu_pct_mean", "disk_write_MBps"]
    hdr = ("| model | sys | " + " | ".join(cols) + " |")
    sep = "|" + "---|" * (len(cols) + 2)
    lines += [hdr, sep]
    for fam in ["llama-3.2-3B", "Qwen2.5-7B", "gemma-2-9B", "other"]:
        sub = med[med.family == fam]
        for model in sorted(sub.model_id.unique()):
            for system in ("tensordex", "tensordex_c",
                           "tensordex_srv_cold", "tensordex_srv_warm"):
                r = sub[(sub.model_id == model) & (sub.system == system)]
                if r.empty:
                    continue
                r = r.iloc[0]
                vals = []
                for c in cols:
                    v = r.get(c)
                    vals.append("" if pd.isna(v) else
                                f"{v:,.0f}" if c in ("gets", "decode_cpu_ms",
                                                     "resolve_ms")
                                else f"{v:.2f}")
                lines.append(f"| {model} | {system.replace('tensordex', 'td')}"
                             f" | " + " | ".join(vals) + " |")
    # Family-level speed comparison.
    lines.append("\n## Family aggregates (median of per-run values)\n")
    fam_med = per_run.groupby(["family", "system"]).median(
        numeric_only=True).reset_index()
    lines.append("| family | sys | e2e_s | s3_GB | net_MBps | "
                 "goodput_MBps | decode_cpu_ms |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in fam_med.iterrows():
        lines.append(
            f"| {r.family} | {r.system.replace('tensordex', 'td')} | "
            f"{r.e2e_s:.1f} | {r.s3_GB:.2f} | {r.net_MBps:.0f} | "
            f"{r.goodput_MBps:.0f} | {r.get('decode_cpu_ms', 0):,.0f} |")

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
