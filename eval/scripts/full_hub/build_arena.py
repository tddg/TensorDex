#!/usr/bin/env python3
"""Fill the 742k x 2048 i32 fingerprint arena from node backup DBs.

Arena row order = ordered_tensors.jsonl (arrival order). Rows not covered by
any node DB are left zero and listed in need_sketch.jsonl for the sketch pass.
"""
import json
import pickle
import sqlite3
import time

import numpy as np

PLAN = "/mnt/nvme0/campaign/plan"
K = 2048

t0 = time.time()
ordered = []
with open(f"{PLAN}/ordered_tensors.jsonl") as f:
    for line in f:
        ordered.append(json.loads(line))
tid2row = {r[0]: i for i, r in enumerate(ordered)}
n = len(ordered)
print(f"{n} tensors ({time.time()-t0:.0f}s)")

arena = np.lib.format.open_memmap(
    f"{PLAN}/fp_arena.npy", mode="w+", dtype=np.int32, shape=(n, K))
filled = np.zeros(n, dtype=bool)

for node in range(16):
    con = sqlite3.connect(f"file:/mnt/nvme0/node{node}.db?mode=ro", uri=True)
    got = 0
    for tid, fp in con.execute(
            "SELECT id, fingerprint FROM tensors WHERE fingerprint IS NOT NULL"):
        row = tid2row.get(tid)
        if row is None or filled[row]:
            continue
        if fp is None or len(fp) != K * 4:
            continue
        arena[row] = np.frombuffer(fp, dtype="<i4")
        filled[row] = True
        got += 1
    con.close()
    print(f"node{node}: +{got} (total {filled.sum()}) {time.time()-t0:.0f}s",
          flush=True)

arena.flush()
np.save(f"{PLAN}/fp_filled.npy", filled)
with open(f"{PLAN}/tid2row.pkl", "wb") as f:
    pickle.dump(tid2row, f)
need = [ordered[i] for i in np.flatnonzero(~filled)]
with open(f"{PLAN}/need_sketch.jsonl", "w") as f:
    for r in need:
        f.write(json.dumps(r) + "\n")
sz = sum(r[3] for r in need)
print(f"DONE filled={int(filled.sum())}/{n}  need_sketch={len(need)} "
      f"({sz/1e12:.2f} TB)  {time.time()-t0:.0f}s")
