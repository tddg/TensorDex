#!/usr/bin/env python3
"""Stage 0 metadata inspection (plan §8 Stage 0).

Dumps tables, schemas, row counts, and small samples from the AE caches:
    ae/cache/data/tensordb_s3/metadata.db   (tensor sizes + model->tensor map)
    ae/cache/results.db                     (compression cache)

Writes a machine-readable summary to eval/raw/metadata_schema.json and a
human-readable report to stdout. Run after `make ae-cache`.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys

DBS = {
    "metadata": "ae/cache/data/tensordb_s3/metadata.db",
    "results": "ae/cache/results.db",
}


def inspect(path: str, sample_rows: int, count_rows: bool) -> dict:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    cur = conn.cursor()
    out = {"path": path, "bytes": os.path.getsize(path), "tables": {}}
    cur.execute("SELECT name, sql FROM sqlite_master "
                "WHERE type IN ('table','index','view') ORDER BY name")
    schema = cur.fetchall()
    for name, sql in schema:
        if sql is None:
            continue
        entry = {"sql": sql}
        try:
            cur.execute(f"PRAGMA table_info({name})")
            cols = cur.fetchall()
            if cols:
                entry["columns"] = [c[1] for c in cols]
                if count_rows:
                    cur.execute(f"SELECT COUNT(*) FROM {name}")
                    entry["rows"] = cur.fetchone()[0]
                if sample_rows:
                    cur.execute(f"SELECT * FROM {name} LIMIT {sample_rows}")
                    entry["sample"] = [
                        [str(v)[:80] for v in r] for r in cur.fetchall()]
        except sqlite3.Error as e:
            entry["error"] = str(e)
        out["tables"][name] = entry
    conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=3)
    ap.add_argument("--no-count", action="store_true",
                    help="skip COUNT(*) (slow on the 5.6 GB results.db)")
    ap.add_argument("--out", default="eval/raw/metadata_schema.json")
    args = ap.parse_args()

    report = {}
    for label, path in DBS.items():
        if not os.path.exists(path):
            print(f"[skip] {label}: {path} not found (run `make ae-cache`)")
            continue
        print(f"\n=== {label}: {path} "
              f"({os.path.getsize(path) / 1e9:.2f} GB) ===")
        info = inspect(path, args.sample_rows, not args.no_count)
        report[label] = info
        for tname, t in info["tables"].items():
            rows = t.get("rows", "?")
            print(f"\n-- {tname} ({rows} rows)")
            print(t["sql"])
            for r in t.get("sample", []):
                print("   ", r)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
