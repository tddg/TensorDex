#!/usr/bin/env python3
"""Plan v2: adopt the paper's FlexSplit plan (AE cache
compression_data/real_compression_all_models.csv) onto the surviving corpus.

Rules (in order):
1. Candidate pairs = their (target, base) rows where BOTH blobs exist
   physically. Apply the AE's own conflict resolution
   (_sim_overall_reduction): a tensor that is both target and base becomes a
   raw base iff its value as a base exceeds its gain as a target; targets
   whose base ended up a target itself are re-rooted... (AE demotes them to
   raw; we instead re-root to the base's base is NOT allowed — depth-1 — so
   we follow AE exactly: they stay raw bases).
2. Fallback: physical tensors with no adopted assignment (orphaned base,
   absent from their CSV, or demoted by rule 1) take our v1 incremental
   plan's pair IF that base is a raw tensor under the final roles (keeps
   depth-1). Provenance recorded per pair.

Output: plan/plan.npz (same schema as v1 + provenance + their_tratio) and a
summary with the expected reduction (their tratio where measured, their
pred_ratio else our v1 pred, else group median).
"""
import csv
import json
import pickle
import time
from collections import defaultdict

import numpy as np

PLAN = "/mnt/nvme0/campaign/plan"
CSV = "/mnt/nvme0/ae_cache/compression_data/real_compression_all_models.csv"


def main():
    t0 = time.time()
    ordered = [json.loads(l) for l in open(f"{PLAN}/ordered_tensors.jsonl")]
    tid2row = {r[0]: i for i, r in enumerate(ordered)}
    sizes = np.array([r[3] for r in ordered], dtype=np.int64)
    existing = pickle.load(open(f"{PLAN}/existing_blobs.pkl", "rb"))
    phys = set(t for t in tid2row if t in existing)

    theirs = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            t, b = r["target_id"], r["base_id"]
            if t == b:
                continue
            bi = int(r["bytes_in"] or 0)
            if bi <= 0:
                continue
            tb = int(r.get("tbytes_out") or 0)
            pr = r.get("pred_ratio")
            theirs[t] = (b, bi, tb, float(pr) if pr else -1.0)
    print(f"their rows: {len(theirs)} ({time.time()-t0:.0f}s)")

    # candidate adopted pairs: both physical
    cand = {t: v for t, v in theirs.items()
            if t in phys and v[0] in phys}

    # AE conflict resolution (verbatim semantics from _sim_overall_reduction)
    children = defaultdict(list)
    for t, (b, _bi, _tb, _pr) in cand.items():
        children[b].append(t)

    def est_cr(v):
        b, bi, tb, pr = v
        if tb > 0:
            return tb / bi
        if 0.0 <= pr <= 1.0:
            return pr
        return 0.35  # their trace-wide weighted mean

    gains = {t: max(0, sizes[tid2row[t]] * (1 - est_cr(v)))
             for t, v in cand.items()}
    base_val = {b: sum(gains.get(k, 0) for k in kids)
                for b, kids in children.items()}
    conflicts = set(cand) & set(children)
    forced = {n for n in conflicts if base_val.get(n, 0) > gains.get(n, 0)}

    final = {}      # tid -> (base, provenance, their_tratio_or_-1)
    roles = {}
    for t, v in cand.items():
        b = v[0]
        if t in forced:
            roles[t] = "base"
            continue
        if b in conflicts and b not in forced:
            roles[t] = "base"
            continue
        final[t] = (b, "paper", v[2] / v[1] if v[2] > 0 else -1.0)
        roles[t] = "target"
        if b not in roles:
            roles[b] = "base"
    n_paper = len(final)

    # fallback from v1 for unassigned physical tensors
    v1 = np.load(f"{PLAN}/plan_v1_incremental.npz")
    v1_pred = {}
    row2tid = [r[0] for r in ordered]
    v1_pair = {}
    for t_r, b_r, p in zip(v1["target_row"].tolist(), v1["base_row"].tolist(),
                           v1["pred_cr"].tolist()):
        v1_pair[row2tid[t_r]] = (row2tid[b_r], p)
        v1_pred[row2tid[t_r]] = p
    n_fb = 0
    for t in phys:
        if roles.get(t) is not None:
            continue
        got = v1_pair.get(t)
        if got is None:
            roles[t] = "base"
            continue
        b, p = got
        if b in phys and roles.get(b, "base") == "base" and b not in final:
            final[t] = (b, "v1", -1.0)
            roles[t] = "target"
            roles.setdefault(b, "base")
            n_fb += 1
        else:
            roles[t] = "base"

    # emit plan.npz
    tr, br, prov, ttr, pred = [], [], [], [], []
    for t, (b, pv, trat) in final.items():
        tr.append(tid2row[t])
        br.append(tid2row[b])
        prov.append(1 if pv == "paper" else 2)
        ttr.append(trat)
        if trat > 0:
            pred.append(trat)
        elif pv == "paper":
            e = theirs[t]
            pred.append(e[3] if 0 <= e[3] <= 1 else 0.35)
        else:
            pred.append(v1_pred.get(t, 0.55))
    tr = np.array(tr, dtype=np.int64)
    br = np.array(br, dtype=np.int64)
    pred = np.array(pred)
    bases = sorted({int(b) for b in br.tolist()})
    np.savez(f"{PLAN}/plan.npz", target_row=tr, base_row=br,
             dist=np.full(len(tr), -1.0), pred_cr=pred,
             provenance=np.array(prov, dtype=np.int8),
             their_tratio=np.array(ttr),
             bases=np.array(bases, dtype=np.int64),
             skipped_no_fp=np.array([], dtype=np.int64))

    tot = int(sizes[[tid2row[t] for t in phys]].sum())
    t_bytes = int(sizes[tr].sum())
    exp_stored = (tot - t_bytes) + float((sizes[tr] * pred).sum())
    print(f"plan v2: {len(tr)} pairs (paper {n_paper}, v1-fallback {n_fb}); "
          f"raw tensors {len(phys)-len(tr)}")
    print(f"targets {t_bytes/1e12:.2f} TB of {tot/1e12:.2f} TB")
    print(f"EXPECTED stored {exp_stored/1e12:.2f} TB -> reduction "
          f"{1-exp_stored/tot:.1%} (their tratio where measured, "
          f"pred/imputed elsewhere)")
    print(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
