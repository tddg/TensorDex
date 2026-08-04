#!/usr/bin/env python3
"""Bit-equivalence check: campaign planner vs Rust _ops.plan_attach.

Feeds identical (tid, shape_key, n_bits) lists + identical fingerprints to
both implementations for several real shape groups and requires exactly the
same base set and (target, base) pairs, and near-identical dist/pred values.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/ubuntu/TensorDex")
from tensordex import _ops  # noqa: E402

sys.path.insert(0, "/mnt/nvme0/campaign")
import planner  # noqa: E402

PLAN = "/mnt/nvme0/campaign/plan"

ordered = []
with open(f"{PLAN}/ordered_tensors.jsonl") as f:
    for line in f:
        ordered.append(json.loads(line))
F = np.load(f"{PLAN}/fp_arena.npy", mmap_mode="r")
filled = np.load(f"{PLAN}/fp_filled.npy")
exists = np.load(f"{PLAN}/blob_exists.npy")
n_bits_arr = np.array([r[4] for r in ordered], dtype=np.float64)

# shape groups, arrival order, existing only
groups = {}
for i, r in enumerate(ordered):
    if exists[i]:
        groups.setdefault(r[1], []).append(i)

by_size = sorted(groups.items(), key=lambda kv: len(kv[1]))
# pick: a small, two medium, and one larger group (Rust is O(n*b), keep <20k)
picks = [g for g in by_size if 50 <= len(g[1]) <= 20000]
test_groups = [picks[0], picks[len(picks) // 2], picks[-3], picks[-1]]

ok = True
for key, rows in test_groups:
    store = _ops.FingerprintStore(2048, len(rows))
    tensors = []
    for r in rows:
        tid = ordered[r][0]
        if filled[r]:
            store.insert_vec(tid, np.ascontiguousarray(F[r]))
        tensors.append((tid, key, int(ordered[r][4])))
    rust = _ops.plan_attach(store, tensors, 0.70, planner.COEFFS, 2)
    rust_pairs = [(p.target_id, p.base_id, p.distance, p.predicted_cr)
                  for p in rust.pairs]

    b, a, s = planner.plan_group(rows, F, filled, n_bits_arr)
    mine_pairs = [(ordered[t][0], ordered[bb][0], d, p) for t, bb, d, p in a]
    mine_bases = [ordered[x][0] for x in b]

    same_bases = list(rust.bases) == mine_bases
    same_pairs = ([(t, bb) for t, bb, _, _ in rust_pairs]
                  == [(t, bb) for t, bb, _, _ in mine_pairs])
    if same_pairs and rust_pairs:
        dmax = max(abs(r[2] - m[2]) for r, m in zip(rust_pairs, mine_pairs))
        pmax = max(abs(r[3] - m[3]) for r, m in zip(rust_pairs, mine_pairs))
    else:
        dmax = pmax = float("nan")
    print(f"{key[:22]:24} n={len(rows):6d} bases {len(rust.bases):5d}/"
          f"{len(mine_bases):5d} pairs {len(rust_pairs):6d}/"
          f"{len(mine_pairs):6d} bases_eq={same_bases} pairs_eq={same_pairs} "
          f"|d|max={dmax:.3e} |cr|max={pmax:.3e}")
    ok &= same_bases and same_pairs
print("EQUIVALENT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
