#!/usr/bin/env python3
"""End-to-end model verification against the hybrid compressed layout.

Samples N random fully-servable models, reconstructs every unique tensor:
  - raw tensor (no tensor_deltas row): GET tensordb/ blob, XXH3 == tid
  - delta tensor: GET compressed_full/ blob + its base from tensordb/,
    decompress, XXH3 == tid
Byte-exactness = content hash equality (the same guarantee `pull --verify`
gives). Prints per-model read-amp (bytes fetched / logical) as a bonus.

Usage: verify_models.py [N] [seed]
"""
import json
import random
import sqlite3
import struct
import sys
import time

sys.path.insert(0, "/home/ubuntu/TensorDex")

import boto3  # noqa: E402
from botocore.config import Config  # noqa: E402
from tensordex import _ops  # noqa: E402

DB = "/mnt/nvme0/campaign/master_compressed.db"
BUCKET = "tensor-tingfeng"


def parse_blob(data):
    hlen = struct.unpack("<Q", data[:8])[0]
    hdr = json.loads(data[8 : 8 + hlen])
    meta = hdr.get("__metadata__", {}).get("tensor")
    key = next(k for k in hdr if k != "__metadata__")
    o0, o1 = hdr[key]["data_offsets"]
    payload = bytes(data[8 + hlen + o0 : 8 + hlen + o1])
    codec = None
    if meta:
        try:
            m = json.loads(meta)
            if isinstance(m, dict) and m.get("codec"):
                codec = m
        except Exception:  # noqa: BLE001
            pass
    return payload, codec


def main():
    n_models = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    s3 = boto3.client("s3", config=Config(
        max_pool_connections=16, retries={"mode": "standard",
                                          "max_attempts": 5}))
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)

    servable = [m for (m,) in con.execute(
        """SELECT model_name FROM model_meta mm WHERE status='ready'
           AND NOT EXISTS (
             SELECT 1 FROM model_mappings g JOIN tensors t ON t.id=g.tensor_id
             WHERE g.model_name = mm.model_name
               AND t.storage_uri LIKE 'missing://%')""")]
    print(f"servable models: {len(servable)}")
    random.Random(seed).shuffle(servable)
    picks = servable[:n_models]

    for model in picks:
        rows = con.execute(
            """SELECT DISTINCT t.id, t.storage_uri, t.logical_bytes,
                      d.base_tensor_id
               FROM model_mappings g JOIN tensors t ON t.id = g.tensor_id
               LEFT JOIN tensor_deltas d ON d.tensor_id = t.id
               WHERE g.model_name = ?""", (model,)).fetchall()
        t0 = time.time()
        fetched = 0
        logical = 0
        n_delta = 0
        base_cache = {}
        for tid, uri, lbytes, base_tid in rows:
            logical += lbytes
            key = uri.split(f"{BUCKET}/")[-1]
            data = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            fetched += len(data)
            payload, codec = parse_blob(data)
            if codec is None:
                if _ops.content_hash(payload) != tid:
                    raise SystemExit(f"RAW HASH MISMATCH {model} {tid}")
            else:
                n_delta += 1
                assert codec["base_tensor_id"] == base_tid
                if base_tid not in base_cache:
                    bkey = f"tensordb/blobs/{base_tid[:2]}/{base_tid}.safetensors"
                    bdata = s3.get_object(Bucket=BUCKET, Key=bkey)["Body"].read()
                    fetched += len(bdata)
                    bp, _ = parse_blob(bdata)
                    if _ops.content_hash(bp) != base_tid:
                        raise SystemExit(f"BASE HASH MISMATCH {base_tid}")
                    base_cache[base_tid] = bp
                rt = bytes(_ops.decompress_tensorx_rust(
                    payload, base_cache[base_tid], int(codec["item_size"])))
                if _ops.content_hash(rt) != tid:
                    raise SystemExit(f"DELTA HASH MISMATCH {model} {tid}")
        print(f"OK {model}: {len(rows)} tensors ({n_delta} deltas) "
              f"logical {logical/1e9:.2f} GB, fetched {fetched/1e9:.2f} GB "
              f"(amp {fetched/max(logical,1):.2f}) {time.time()-t0:.0f}s",
              flush=True)
    print("ALL MODELS BYTE-EXACT")


if __name__ == "__main__":
    main()
