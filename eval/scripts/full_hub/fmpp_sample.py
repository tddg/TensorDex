#!/usr/bin/env python3
"""FM++ sampled fratio validation for the 70.5% claim.

Stratified sample of plan-v2 pairs by target-size decile:
  - ~400 pairs that HAVE a published fratio in results.db (cross-validation:
    our on-box FM++ bytes vs theirs, same (target, base))
  - ~100 pairs with NO published fratio (coverage extension)
Every sampled pair is also roundtrip-verified (decompress == target bytes).
Writes logs/fmpp_sample.jsonl.
"""
import json
import pickle
import random
import sqlite3
import struct
import sys
import time

import numpy as np

sys.path.insert(0, "/home/ubuntu/TensorDex")
import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402
from tensordex import _ops  # noqa: E402

PLAN = "/mnt/nvme0/campaign/plan"
LOGS = "/mnt/nvme0/campaign/logs"
BUCKET = "tensor-tingfeng"
N_COVERED = 400
N_UNCOVERED = 100
SEED = 0


def parse_blob(data):
    hlen = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8 : 8 + hlen])
    key = next(k for k in hdr if k != "__metadata__")
    o0, o1 = hdr[key]["data_offsets"]
    return bytes(data[8 + hlen + o0 : 8 + hlen + o1])


def main():
    ordered = [json.loads(l) for l in open(f"{PLAN}/ordered_tensors.jsonl")]
    row2 = [(r[0], r[3], r[2]) for r in ordered]  # tid, size, dtype
    plan = np.load(f"{PLAN}/plan.npz")
    tr, br = plan["target_row"].tolist(), plan["base_row"].tolist()

    con = sqlite3.connect("file:/mnt/nvme0/ae_cache/results.db?mode=ro",
                          uri=True)
    fr = {}
    for t, b, fb, bi in con.execute(
            "SELECT target_id, base_id, fbytes_out, bytes_in "
            "FROM compression_results WHERE fbytes_out > 0 AND bytes_in > 0"):
        fr[(t, b)] = (int(fb), int(bi))

    pairs = []
    for t, b in zip(tr, br):
        tid, sz, dt = row2[t]
        btid = row2[b][0]
        if dt != row2[b][2]:
            continue
        pairs.append((tid, btid, sz, (tid, btid) in fr))
    covered = [p for p in pairs if p[3]]
    uncovered = [p for p in pairs if not p[3]]
    print(f"pairs: {len(pairs)} (covered {len(covered)})")

    rng = random.Random(SEED)

    def stratified(pool, n):
        pool = sorted(pool, key=lambda p: p[2])
        step = max(1, len(pool) // 10)
        deciles = [pool[i : i + step] for i in range(0, len(pool), step)][:10]
        per = max(1, n // 10)
        out = []
        for d in deciles:
            d = list(d)
            rng.shuffle(d)
            out += d[:per]
        return out

    only_uncovered = "--uncovered-only" in sys.argv
    if only_uncovered:
        sample = stratified(uncovered, N_UNCOVERED)
    else:
        sample = (stratified(covered, N_COVERED)
                  + stratified(uncovered, N_UNCOVERED))
    print(f"sampled {len(sample)} pairs")

    s3 = boto3.client("s3", config=Config(
        max_pool_connections=8, retries={"mode": "standard",
                                         "max_attempts": 5}))
    out = open(f"{LOGS}/fmpp_sample.jsonl",
               "a" if only_uncovered else "w", buffering=1)
    n_ok = n_match = n_rt = 0
    sum_in = sum_out = 0
    t0 = time.time()
    for i, (tid, btid, sz, has_ref) in enumerate(sample):
        try:
            tp = parse_blob(s3.get_object(
                Bucket=BUCKET,
                Key=f"tensordb/blobs/{tid[:2]}/{tid}.safetensors")["Body"].read())
            bp = parse_blob(s3.get_object(
                Bucket=BUCKET,
                Key=f"tensordb/blobs/{btid[:2]}/{btid}.safetensors")["Body"].read())
            comp = bytes(_ops.compress_fmpp_rust(tp, bp, 2))
            rt = bytes(_ops.decompress_fmpp_rust(comp, bp, 2))
            rt_ok = _ops.content_hash(rt) == tid
            rec = {"tid": tid, "base": btid, "orig": len(tp),
                   "fmpp": len(comp), "roundtrip_ok": rt_ok}
            if has_ref:
                fb, fbi = fr[(tid, btid)]
                rec["ref_fmpp"] = fb
                rec["ref_in"] = fbi
                if fbi == len(tp):
                    n_match += abs(fb - len(comp)) <= max(64, 0.001 * fb)
            n_ok += 1
            n_rt += rt_ok
            sum_in += len(tp)
            sum_out += len(comp)
            out.write(json.dumps(rec) + "\n")
        except Exception as exc:  # noqa: BLE001
            out.write(json.dumps({"tid": tid, "error": repr(exc)}) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sample)} ok={n_ok} rt_ok={n_rt} "
                  f"wfratio={sum_out/max(sum_in,1):.3f} "
                  f"{time.time()-t0:.0f}s", flush=True)
    print(f"DONE ok={n_ok} roundtrip_ok={n_rt} "
          f"ref-match(within 0.1%)={n_match} "
          f"sample weighted fratio {sum_out/max(sum_in,1):.3f} "
          f"({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
