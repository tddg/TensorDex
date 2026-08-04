#!/usr/bin/env python3
"""Campaign analysis: realized vs predicted CRs, cumulative reduction curve in
model-arrival order (paper Fig 11a comparable), per-family stats, throughput
and bottleneck breakdown from per-pair timings.

Reads logs/exec_done_*.jsonl + plan/plan.npz + ordered_tensors.jsonl.
Writes plan/analysis.json + prints a summary. Safe to run mid-campaign
(reports on completed pairs only).
"""
import glob
import json
import time

import numpy as np

PLAN = "/mnt/nvme0/campaign/plan"
LOGS = "/mnt/nvme0/campaign/logs"


def main():
    t0 = time.time()
    ordered = [json.loads(l) for l in open(f"{PLAN}/ordered_tensors.jsonl")]
    tid2row = {r[0]: i for i, r in enumerate(ordered)}
    sizes = np.array([r[3] for r in ordered], dtype=np.int64)
    plan = np.load(f"{PLAN}/plan.npz")
    tr = plan["target_row"]
    pred = plan["pred_cr"]
    pred_by_row = {}
    for t, p in zip(tr.tolist(), pred.tolist()):
        pred_by_row[t] = p

    comp_by_row = {}
    timings = {"fetch": [], "compress": [], "verify": [], "upload": []}
    skipped = []
    for path in glob.glob(f"{LOGS}/exec_done_*.jsonl"):
        for line in open(path):
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("status") == "ok":
                row = tid2row[r["tid"]]
                comp_by_row[row] = (r["orig"], r["comp"], r["blob"])
                timings["fetch"].append(r["fetch_s"])
                timings["compress"].append(r["compress_s"])
                timings["verify"].append(r["verify_s"])
                timings["upload"].append(r["upload_s"])
            elif r.get("status") == "skipped":
                skipped.append(r["tid"])

    n = len(comp_by_row)
    if not n:
        print("no completed pairs yet")
        return
    orig = np.array([v[0] for v in comp_by_row.values()], dtype=np.int64)
    comp = np.array([v[1] for v in comp_by_row.values()], dtype=np.int64)
    cr = comp / orig
    predv = np.array([pred_by_row.get(r, np.nan) for r in comp_by_row])

    print(f"pairs done {n}/{len(tr)}  ({time.time()-t0:.0f}s to load)")
    print(f"bytes in {orig.sum()/1e12:.2f} TB -> {comp.sum()/1e12:.2f} TB "
          f"(weighted CR {comp.sum()/orig.sum():.3f})")
    print(f"CR percentiles: p10 {np.percentile(cr,10):.3f} "
          f"p50 {np.percentile(cr,50):.3f} p90 {np.percentile(cr,90):.3f} "
          f"mean {cr.mean():.3f}")
    m = ~np.isnan(predv)
    if m.any():
        err = cr[m] - predv[m]
        print(f"pred_cr error: mean {err.mean():+.3f} MAE "
              f"{np.abs(err).mean():.3f} (n={int(m.sum())})")
    for k, v in timings.items():
        a = np.array(v)
        print(f"  {k:9} p50 {np.percentile(a,50)*1e3:6.0f} ms  "
              f"p95 {np.percentile(a,95)*1e3:7.0f} ms  sum {a.sum()/3600:.1f} "
              f"core-h")

    # cumulative reduction in arrival order over the physical corpus,
    # counting done pairs at realized CR, pending at predicted, bases raw
    exists = np.load(f"{PLAN}/blob_exists.npy")
    stored = np.where(exists, sizes.astype(np.float64), 0.0)
    for t, p in zip(tr.tolist(), pred.tolist()):
        stored[t] = sizes[t] * p
    realized_only = 0
    for row, (o, c, _b) in comp_by_row.items():
        stored[row] = c
    logical = np.where(exists, sizes, 0).astype(np.float64)
    cum_logical = np.cumsum(logical)
    cum_stored = np.cumsum(stored)
    red = 1 - cum_stored[-1] / cum_logical[-1]
    print(f"\ncumulative reduction (done@realized, pending@predicted): "
          f"{red:.1%}")
    ck = np.linspace(0, len(sizes) - 1, 25).astype(int)
    np.savez(f"{PLAN}/analysis_curve.npz", rows=ck,
             logical=cum_logical[ck], stored=cum_stored[ck])
    with open(f"{PLAN}/analysis.json", "w") as f:
        json.dump({
            "pairs_done": int(n), "pairs_total": int(len(tr)),
            "bytes_in": int(orig.sum()), "bytes_out": int(comp.sum()),
            "weighted_cr": float(comp.sum() / orig.sum()),
            "cr_p50": float(np.percentile(cr, 50)),
            "reduction_current_estimate": float(red),
            "skipped": len(skipped),
        }, f, indent=2)


if __name__ == "__main__":
    main()
