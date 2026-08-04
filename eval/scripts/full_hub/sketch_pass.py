#!/usr/bin/env python3
"""Partial sketch pass: compute d=2,w=1024 BCS fingerprints for tensors not
covered by node backup DBs, writing rows directly into fp_arena.npy.

Resumable: per-shard done-logs; rerun skips finished tids. Missing or
hash-mismatching blobs land in errors.jsonl and stay unfilled (planner will
treat them as no-fp bases; the exec pass excludes missing blobs entirely).
"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import queue
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/TensorDex")

PLAN = "/mnt/nvme0/campaign/plan"
LOGS = "/mnt/nvme0/campaign/logs"
BUCKET = "tensor-tingfeng"
K = 2048
NPROC = 24
FETCH_THREADS = 4


def worker(shard_id: int, items: list) -> None:
    import boto3
    from botocore.config import Config
    from tensordex import _ops

    s3 = boto3.client("s3", config=Config(
        max_pool_connections=FETCH_THREADS + 2,
        retries={"mode": "standard", "max_attempts": 5}))
    arena = np.lib.format.open_memmap(f"{PLAN}/fp_arena.npy", mode="r+")
    done_path = f"{LOGS}/sketch_done_{shard_id}.log"
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            done = set(line.strip() for line in f)
    todo = [it for it in items if it[0] not in done]
    donef = open(done_path, "a", buffering=1)
    errf = open(f"{LOGS}/sketch_errors_{shard_id}.jsonl", "a", buffering=1)

    q: "queue.Queue" = queue.Queue(maxsize=FETCH_THREADS * 2)

    def fetch(item):
        tid = item[0]
        try:
            body = s3.get_object(
                Bucket=BUCKET,
                Key=f"tensordb/blobs/{tid[:2]}/{tid}.safetensors")["Body"]
            data = body.read()
        except Exception as exc:  # noqa: BLE001
            return (item, None, repr(exc))
        return (item, data, None)

    def producer():
        with cf.ThreadPoolExecutor(FETCH_THREADS) as pool:
            for res in pool.map(fetch, todo):
                q.put(res)
        q.put(None)

    import threading
    threading.Thread(target=producer, daemon=True).start()

    nbytes = 0
    nok = nerr = 0
    t0 = time.time()
    while True:
        res = q.get()
        if res is None:
            break
        (tid, _shape, _dtype, _sz, row), data, err = (
            (res[0][0], res[0][1], res[0][2], res[0][3], res[0][4]),
            res[1], res[2])
        if err is not None:
            errf.write(json.dumps({"tid": tid, "error": err}) + "\n")
            nerr += 1
            continue
        try:
            hlen = struct.unpack("<Q", data[:8])[0]
            hdr = json.loads(data[8 : 8 + hlen])
            key = next(k for k in hdr if k != "__metadata__")
            o0, o1 = hdr[key]["data_offsets"]
            payload = data[8 + hlen + o0 : 8 + hlen + o1]
            if _ops.content_hash(bytes(payload)) != tid:
                raise ValueError("content hash mismatch")
            u16 = np.frombuffer(payload, dtype=np.uint16)
            fp = np.asarray(
                _ops.compute_bcs_fingerprint_u16_py(u16, 2, 1024),
                dtype=np.int32)
            arena[row] = fp
            donef.write(tid + "\n")
            nok += 1
            nbytes += len(payload)
        except Exception as exc:  # noqa: BLE001
            errf.write(json.dumps({"tid": tid, "error": repr(exc)}) + "\n")
            nerr += 1
        if (nok + nerr) % 200 == 0:
            dt = time.time() - t0
            print(f"[shard {shard_id}] {nok+nerr}/{len(todo)} "
                  f"{nbytes/1e9:.1f} GB {nbytes/dt/1e6:.0f} MB/s "
                  f"errs={nerr}", flush=True)
    arena.flush()
    print(f"[shard {shard_id}] DONE ok={nok} err={nerr} "
          f"{nbytes/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True)


def main() -> None:
    os.makedirs(LOGS, exist_ok=True)
    items = []
    with open(f"{PLAN}/need_sketch.jsonl") as f:
        for line in f:
            items.append(json.loads(line))
    import pickle
    with open(f"{PLAN}/tid2row.pkl", "rb") as f:
        tid2row = pickle.load(f)
    # attach arena row; sort by size so shards get similar byte budgets
    items = [(t, s, d, z, tid2row[t]) for t, s, d, z, _nb in items]
    items.sort(key=lambda it: -it[3])
    shards = [items[i::NPROC] for i in range(NPROC)]
    total = sum(it[3] for it in items)
    print(f"{len(items)} tensors, {total/1e12:.2f} TB, {NPROC} procs")

    procs = []
    for i, shard in enumerate(shards):
        p = mp.Process(target=worker, args=(i, shard))
        p.start()
        procs.append(p)
    rc = 0
    for p in procs:
        p.join()
        rc |= p.exitcode or 0
    print("ALL DONE rc=", rc)
    sys.exit(rc)


if __name__ == "__main__":
    main()
