#!/usr/bin/env python3
"""Execute the global FlexSplit plan: TensorX-compress every attached pair and
upload delta blobs to s3://tensor-tingfeng/compressed_full/.

Never touches tensordb/ (read-only source) or compressed_eval/.

Work unit = (base_row, [target_rows...]) with total target bytes capped at
UNIT_TARGET_BYTES; oversized base groups split into multiple units (base
re-fetched per unit — counted in stats). ~NPROC worker processes, each:
  GET base -> verify hash -> for each target: GET -> verify hash ->
  compress_tensorx (level 1) -> decompress -> XXH3 == tid (byte-exact) ->
  wrap in the codec.py safetensors delta format -> PUT to compressed_full.

Resume: per-worker done-logs keyed by target tid; rerun skips completed.
Per-pair records: actual compressed size, timings for fetch/compress/verify/
upload -> logs/exec_done_*.jsonl (the FAST'27 per-pair dataset).
"""
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import queue
import struct
import sys
import threading
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/TensorDex")

PLAN = "/mnt/nvme0/campaign/plan"
LOGS = "/mnt/nvme0/campaign/logs"
BUCKET = "tensor-tingfeng"
SRC = "tensordb"
DST = "compressed_full"
LEVEL = 1
NPROC = 28
UNIT_TARGET_BYTES = 4 * 1024**3
FETCH_THREADS = 3          # per worker: prefetch targets while compressing
ITEM_SIZE = {"torch.bfloat16": 2, "torch.float16": 2, "torch.float32": 4}
ST_DTYPE = {"torch.bfloat16": "BF16", "torch.float16": "F16",
            "torch.float32": "F32"}


def parse_blob(data: bytes):
    hlen = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8 : 8 + hlen])
    key = next(k for k in hdr if k != "__metadata__")
    o0, o1 = hdr[key]["data_offsets"]
    return bytes(data[8 + hlen + o0 : 8 + hlen + o1])


def wrap_delta_blob(tid, base_tid, item_size, target_shape, target_dtype,
                    payload: bytes) -> bytes:
    """Safetensors blob in the exact core/codec.py compressed format."""
    meta = {
        "codec": "tensorx",
        "base_tensor_id": base_tid,
        "item_size": int(item_size),
        "level": LEVEL,
        "target_shape": [int(x) for x in target_shape],
        "target_dtype": target_dtype.removeprefix("torch."),
    }
    hdr = {
        "__metadata__": {"tensor": json.dumps(meta, separators=(",", ":"))},
        "tensor": {"dtype": "U8", "shape": [len(payload)],
                   "data_offsets": [0, len(payload)]},
    }
    hjson = json.dumps(hdr, separators=(",", ":")).encode()
    pad = (8 - (8 + len(hjson)) % 8) % 8
    hjson += b" " * pad
    return struct.pack("<Q", len(hjson)) + hjson + payload


def worker(shard_id: int, units: list, meta: dict) -> None:
    import boto3
    from botocore.config import Config
    from tensordex import _ops

    s3 = boto3.client("s3", config=Config(
        max_pool_connections=FETCH_THREADS + 4,
        retries={"mode": "standard", "max_attempts": 6}))

    done_path = f"{LOGS}/exec_done_{shard_id}.jsonl"
    done = set()
    if os.path.exists(done_path):
        with open(done_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["tid"])
                except Exception:  # noqa: BLE001
                    pass
    out = open(done_path, "a", buffering=1)
    errf = open(f"{LOGS}/exec_errors_{shard_id}.jsonl", "a", buffering=1)

    def get_payload(tid):
        t0 = time.time()
        body = s3.get_object(
            Bucket=BUCKET, Key=f"{SRC}/blobs/{tid[:2]}/{tid}.safetensors"
        )["Body"]
        data = body.read()
        payload = parse_blob(data)
        if _ops.content_hash(payload) != tid:
            raise ValueError(f"source hash mismatch {tid}")
        return payload, time.time() - t0

    stats = {"pairs": 0, "in": 0, "out": 0, "up": 0, "err": 0}
    t_start = time.time()

    for base_row, target_rows in units:
        base_tid, base_shape, base_dtype, base_sz = meta[base_row]
        todo = [r for r in target_rows if meta[r][0] not in done]
        if not todo:
            continue
        try:
            base_payload, base_dt = get_payload(base_tid)
        except Exception as exc:  # noqa: BLE001
            errf.write(json.dumps({"base": base_tid, "error": repr(exc),
                                   "targets": len(todo)}) + "\n")
            stats["err"] += len(todo)
            continue

        # prefetch targets while the CPU compresses
        q: "queue.Queue" = queue.Queue(maxsize=FETCH_THREADS)

        def fetch_target(row):
            tid = meta[row][0]
            try:
                payload, dt = get_payload(tid)
                return (row, payload, dt, None)
            except Exception as exc:  # noqa: BLE001
                return (row, None, 0.0, repr(exc))

        def producer():
            with cf.ThreadPoolExecutor(FETCH_THREADS) as pool:
                for res in pool.map(fetch_target, todo):
                    q.put(res)
            q.put(None)

        threading.Thread(target=producer, daemon=True).start()

        while True:
            item = q.get()
            if item is None:
                break
            row, payload, fetch_dt, err = item
            tid, shape, dtype, sz = meta[row]
            if err is not None:
                errf.write(json.dumps({"tid": tid, "error": err}) + "\n")
                stats["err"] += 1
                continue
            try:
                if dtype != base_dtype:
                    out.write(json.dumps({
                        "tid": tid, "base": base_tid, "status": "skipped",
                        "reason": "dtype_mismatch"}) + "\n")
                    continue
                isz = ITEM_SIZE[dtype]
                t0 = time.time()
                comp = bytes(_ops.compress_tensorx_rust(
                    payload, base_payload, isz, LEVEL))
                t1 = time.time()
                rt = bytes(_ops.decompress_tensorx_rust(
                    comp, base_payload, isz))
                if _ops.content_hash(rt) != tid:
                    raise ValueError("roundtrip hash mismatch")
                del rt
                t2 = time.time()
                blob = wrap_delta_blob(tid, base_tid, isz, json.loads(shape),
                                       dtype, comp)
                s3.put_object(
                    Bucket=BUCKET,
                    Key=f"{DST}/blobs/{tid[:2]}/{tid}.safetensors",
                    Body=blob)
                t3 = time.time()
                out.write(json.dumps({
                    "tid": tid, "base": base_tid, "status": "ok",
                    "orig": sz, "comp": len(comp), "blob": len(blob),
                    "fetch_s": round(fetch_dt, 3),
                    "compress_s": round(t1 - t0, 3),
                    "verify_s": round(t2 - t1, 3),
                    "upload_s": round(t3 - t2, 3)}) + "\n")
                stats["pairs"] += 1
                stats["in"] += sz
                stats["out"] += len(comp)
                stats["up"] += len(blob)
            except Exception as exc:  # noqa: BLE001
                errf.write(json.dumps({"tid": tid, "base": base_tid,
                                       "error": repr(exc)}) + "\n")
                stats["err"] += 1
        del base_payload
        if stats["pairs"] and stats["pairs"] % 200 < len(todo):
            dt = time.time() - t_start
            print(f"[w{shard_id}] pairs={stats['pairs']} "
                  f"in={stats['in']/1e9:.1f}GB out={stats['out']/1e9:.1f}GB "
                  f"({stats['in']/dt/1e6:.0f} MB/s) errs={stats['err']}",
                  flush=True)

    print(f"[w{shard_id}] DONE {json.dumps(stats)} "
          f"{time.time()-t_start:.0f}s", flush=True)


def main() -> None:
    os.makedirs(LOGS, exist_ok=True)
    t0 = time.time()
    ordered = []
    with open(f"{PLAN}/ordered_tensors.jsonl") as f:
        for line in f:
            ordered.append(json.loads(line))
    plan = np.load(f"{PLAN}/plan.npz")
    tr, br = plan["target_row"], plan["base_row"]

    meta = {}
    for i, (tid, shape, dtype, sz, _nb) in enumerate(ordered):
        meta[i] = (tid, shape, dtype, sz)

    groups = {}
    for t, b in zip(tr.tolist(), br.tolist()):
        groups.setdefault(b, []).append(t)

    # split into work units capped by target bytes
    units = []
    for b, targets in groups.items():
        cur, acc = [], 0
        for t in targets:
            sz = meta[t][3]
            if cur and acc + sz > UNIT_TARGET_BYTES:
                units.append((b, cur))
                cur, acc = [], 0
            cur.append(t)
            acc += sz
        if cur:
            units.append((b, cur))

    # balance shards by unit weight (base + targets)
    def unit_weight(u):
        b, ts = u
        return meta[b][3] + sum(meta[t][3] for t in ts)

    units.sort(key=unit_weight, reverse=True)
    shards = [[] for _ in range(NPROC)]
    weights = [0] * NPROC
    for u in units:
        i = weights.index(min(weights))
        shards[i].append(u)
        weights[i] += unit_weight(u)

    tot_t = sum(meta[t][3] for t in tr.tolist())
    tot_b = sum(meta[b][3] for b in groups)
    refetch = sum(meta[b][3] for b, _ in units) - tot_b
    print(f"{len(tr)} pairs, {len(groups)} base groups, {len(units)} units; "
          f"targets {tot_t/1e12:.2f} TB + bases {tot_b/1e12:.2f} TB "
          f"(+{refetch/1e12:.2f} TB base refetch) ({time.time()-t0:.0f}s)",
          flush=True)

    # slim meta per shard (only rows it needs) to keep fork cheap
    procs = []
    for i, shard in enumerate(shards):
        need = set()
        for b, ts in shard:
            need.add(b)
            need.update(ts)
        m = {r: meta[r] for r in need}
        p = mp.Process(target=worker, args=(i, shard, m))
        p.start()
        procs.append(p)
    rc = 0
    for p in procs:
        p.join()
        rc |= p.exitcode or 0
    print(f"ALL DONE rc={rc} wall={time.time()-t0:.0f}s")
    sys.exit(rc)


if __name__ == "__main__":
    main()
