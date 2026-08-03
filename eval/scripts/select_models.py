#!/usr/bin/env python3
"""Stratified model selection (plan §7) from the AE metadata.db.

Computable from the published metadata (tensors + model_mappings):
  - logical model size, tensor count, family (HF org / name heuristics)
  - cross-model tensor sharing (dedup fan-in) as a fragmentation proxy

NOT computable until the bucket's master.db (with tensor_deltas) is
available: dependency depth, delta-vs-direct mix, base-reuse strata.
Those columns are emitted as null and must be filled before freezing
eval/config/models.yaml.

Usage:
    python eval/scripts/select_models.py \
        [--db ae/cache/data/tensordb_s3/metadata.db] [--per-cell 2]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

import yaml

SIZE_STRATA = [  # (label, lo_pct, hi_pct)
    ("small", 0.10, 0.25),
    ("medium", 0.45, 0.55),
    ("large", 0.75, 0.90),
    ("very_large", 0.95, 1.00),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="ae/cache/data/tensordb_s3/metadata.db")
    ap.add_argument("--per-cell", type=int, default=2,
                    help="models per (size x sharing) cell")
    ap.add_argument("--out", default="eval/config/models.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    con.execute("PRAGMA temp_store=MEMORY")

    print("computing per-model aggregates ...")
    rows = con.execute("""
        SELECT mm.model_name,
               COUNT(*)                       AS n_params,
               COUNT(DISTINCT mm.tensor_id)   AS n_unique,
               SUM(t.size_bytes)              AS logical_bytes,
               AVG(sh.n_models)               AS mean_share,
               MAX(sh.n_models)               AS max_share
        FROM model_mappings mm
        JOIN tensors t ON t.id = mm.tensor_id
        JOIN (SELECT tensor_id, COUNT(DISTINCT model_name) AS n_models
              FROM model_mappings GROUP BY tensor_id) sh
             ON sh.tensor_id = mm.tensor_id
        GROUP BY mm.model_name
    """).fetchall()
    con.close()
    print(f"{len(rows)} models")

    rows.sort(key=lambda r: r[3] or 0)
    n = len(rows)
    import random
    rng = random.Random(args.seed)

    selected = {}
    for label, lo, hi in SIZE_STRATA:
        stratum = rows[int(lo * n):max(int(hi * n), int(lo * n) + 1)]
        # Split by sharing (dedup fan-in) as the fragmentation proxy:
        # low = tensors mostly unique to this model, high = heavily shared.
        stratum = sorted(stratum, key=lambda r: r[4] or 0)
        cells = {
            "low_sharing": stratum[: max(1, len(stratum) // 3)],
            "high_sharing": stratum[-max(1, len(stratum) // 3):],
        }
        for cell, cand in cells.items():
            for r in rng.sample(cand, min(args.per_cell, len(cand))):
                selected[r[0]] = {
                    "size_class": label,
                    "sharing_class": cell,
                    "family": r[0].split("/")[0],
                    "n_params": r[1],
                    "n_unique_tensors": r[2],
                    "logical_bytes": int(r[3] or 0),
                    "mean_models_sharing_tensor": round(r[4] or 0, 2),
                    "max_models_sharing_tensor": int(r[5] or 0),
                    # Filled once master.db / tensor_deltas is available:
                    "dependency_stats": None,
                    "tensordex_objects": None,
                    "zipnn_objects": None,
                    "raw_objects": None,
                }

    with open(args.out, "w") as fh:
        yaml.safe_dump(selected, fh, sort_keys=True)
    for m, v in sorted(selected.items(),
                       key=lambda kv: kv[1]["logical_bytes"]):
        print(f"{v['size_class']:11s} {v['sharing_class']:12s} "
              f"{v['logical_bytes'] / 1e9:8.2f} GB  {v['n_params']:5d}t  {m}")
    print(f"\nwrote {args.out} ({len(selected)} models) — "
          "dependency strata pending master.db")
    return 0


if __name__ == "__main__":
    sys.exit(main())
