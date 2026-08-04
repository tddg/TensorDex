#!/usr/bin/env python3
"""Global FlexSplit Phase-I attach planning over the physical 643k-tensor trace.

Bit-faithful, vectorized reimplementation of `_ops.plan_attach` (planner.rs):
per shape group, tensors arrive in order; each attaches to the nearest opened
base iff predict_cr(dist) <= 0.70, else opens a new base. Distances are exact
integer squared-L2 over BCS fingerprints computed via f64 GEMM (inputs are
integer-valued, every intermediate < 2^53, so results are exact), normalized
by d_rows=2 and the target's n_bits — identical to planner.rs bcs_distance.

Input rows are restricted to tensors whose blob exists in S3; all of them
have fingerprints, but the no-fp path is kept for parity.
"""
import json
import math
import time

import numpy as np

PLAN = "/mnt/nvme0/campaign/plan"
CR_THRESHOLD = 0.70
COEFFS = (-23.727944, 0.522466, 1.966862, -0.043132)  # HYBRID_COEFFS_V3
D_ROWS = 2.0
CHUNK = 4096


def predict_cr(p: float) -> float:
    p = min(max(p, 0.0), 0.5)
    if p <= 1e-15 or p >= 1.0 - 1e-15:
        h = 0.0
    else:
        h = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
    t = 8.0 * h
    a, b, c, d = COEFFS
    return min(max(a * p + b * t + c * p * t + d, 0.0), 1.0)


def plan_group(rows, F, filled, n_bits_arr):
    """rows: arena row indices of one shape group in arrival order.

    Returns (bases, attach, skipped_no_fp); attach entries are
    (target_row, base_row, dist, pred_cr). Mirrors planner.rs exactly:
    - first tensor anchors the group (base even without fp)
    - no-fp tensor -> new base, never attracts members
    - else nearest fp'd base (first-wins ties); attach iff pred_cr <= 0.70
    """
    n = len(rows)
    Gg = F[rows].astype(np.float64)
    sq = np.einsum("ij,ij->i", Gg, Gg)

    bases = []        # local indices of ALL bases
    fpb = []          # local indices of fp'd bases, insertion order
    attach = []
    skipped = []

    i = 0
    while i < n:
        j = min(i + CHUNK, n)
        nb0 = len(fpb)
        if nb0:
            B = np.asarray(fpb, dtype=np.int64)
            D0 = sq[i:j, None] - 2.0 * (Gg[i:j] @ Gg[B].T) + sq[B][None, :]
        else:
            D0 = None
        new_cols = []     # distance columns for bases opened inside chunk

        def open_fp_base(q):
            fpb.append(q)
            new_cols.append(sq[i:j] - 2.0 * (Gg[i:j] @ Gg[q]) + sq[q])

        for q in range(i, j):
            li = q - i
            has_fp = bool(filled[rows[q]])
            if not bases:
                bases.append(q)
                if has_fp:
                    open_fp_base(q)
                else:
                    skipped.append(q)
                continue
            if not has_fp:
                bases.append(q)
                skipped.append(q)
                continue
            if not fpb:
                bases.append(q)
                open_fp_base(q)
                continue
            parts = []
            if nb0:
                parts.append(D0[li, :nb0])
            for col in new_cols:
                parts.append(col[li : li + 1])
            full = np.concatenate(parts) if len(parts) > 1 else parts[0]
            dists = full / D_ROWS / max(float(n_bits_arr[rows[q]]), 1.0)
            bi = int(np.argmin(dists))
            best = float(dists[bi])
            pred = predict_cr(best)
            if pred <= CR_THRESHOLD:
                attach.append((rows[q], rows[fpb[bi]], best, pred))
            else:
                bases.append(q)
                open_fp_base(q)
        i = j

    return ([rows[b] for b in bases], attach, [rows[s] for s in skipped])


def main():
    t0 = time.time()
    ordered = []
    with open(f"{PLAN}/ordered_tensors.jsonl") as f:
        for line in f:
            ordered.append(json.loads(line))
    F = np.load(f"{PLAN}/fp_arena.npy", mmap_mode="r")
    filled = np.load(f"{PLAN}/fp_filled.npy")
    exists = np.load(f"{PLAN}/blob_exists.npy")
    n_bits_arr = np.array([r[4] for r in ordered], dtype=np.float64)
    sizes = np.array([r[3] for r in ordered], dtype=np.int64)

    groups = {}
    order = []
    for i, r in enumerate(ordered):
        if not exists[i]:
            continue
        key = r[1]
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)
    n_in = sum(len(v) for v in groups.values())
    print(f"{n_in} tensors (of {len(ordered)} registered), "
          f"{len(groups)} shape groups ({time.time()-t0:.0f}s)", flush=True)

    all_bases, all_attach, all_skipped = [], [], []
    for key in sorted(order, key=lambda k: -len(groups[k])):
        rows = groups[key]
        gt0 = time.time()
        b, a, s = plan_group(rows, F, filled, n_bits_arr)
        all_bases += b
        all_attach += a
        all_skipped += s
        if len(rows) > 5000:
            print(f"  {key[:24]:26} n={len(rows):7d} "
                  f"({sizes[rows].sum()/1e12:5.2f} TB) bases={len(b):6d} "
                  f"attach={len(a):7d} {time.time()-gt0:6.1f}s", flush=True)

    tr = np.array([a[0] for a in all_attach], dtype=np.int64)
    br = np.array([a[1] for a in all_attach], dtype=np.int64)
    dist = np.array([a[2] for a in all_attach])
    pred = np.array([a[3] for a in all_attach])
    np.savez(f"{PLAN}/plan.npz", target_row=tr, base_row=br, dist=dist,
             pred_cr=pred, bases=np.array(all_bases, dtype=np.int64),
             skipped_no_fp=np.array(all_skipped, dtype=np.int64))

    tot = sizes[np.flatnonzero(exists)].sum()
    attached_bytes = sizes[tr].sum()
    pred_stored = (tot - attached_bytes) + (sizes[tr] * pred).sum()
    print(f"\nPLAN DONE {time.time()-t0:.0f}s")
    print(f"  tensors {n_in}  bases {len(all_bases)}  attached {len(tr)}  "
          f"no-fp {len(all_skipped)}")
    print(f"  bytes: total {tot/1e12:.2f} TB, attached "
          f"{attached_bytes/1e12:.2f} TB ({attached_bytes/tot:.1%})")
    print(f"  PREDICTED stored {pred_stored/1e12:.2f} TB  "
          f"reduction {1 - pred_stored/tot:.1%}")


if __name__ == "__main__":
    main()
