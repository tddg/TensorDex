#!/usr/bin/env python3
"""Upload the compressed hub to s3://tensor-tingfeng/compressed_eval/.

SAFETY CONTRACT (user requirement):
  - WRITE-ONLY: this script performs no delete, copy, or lifecycle
    operations of any kind — only PutObject via upload_file.
  - Every key is asserted to start with the destination prefix before any
    request is issued; anything else aborts the run.
  - The source store under tensordb/ is never touched.

Layout written (mirrors the live store so S3StorageBackend can read it):
  compressed_eval/blobs/{tid[:2]}/{tid}.safetensors   (1-level shard)
  compressed_eval/master.db                            (hub metadata incl.
                                                        tensor_deltas)

Local hub blobs are 2-level (blobs/xx/yy/id.safetensors); keys are
flattened to the 1-level S3 scheme used by S3StorageBackend._blob_key.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import threading
import time

import boto3
from botocore.config import Config

BUCKET = "tensor-tingfeng"
DEST_PREFIX = "compressed_eval/"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="eval/raw/compressed_hub")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    blob_root = os.path.join(args.hub, "blobs")
    meta_path = os.path.join(args.hub, "metadata.db")
    if not (os.path.isdir(blob_root) and os.path.exists(meta_path)):
        print(f"hub incomplete under {args.hub}")
        return 2

    uploads: list[tuple[str, str]] = []  # (local_path, key)
    for dirpath, _dirs, files in os.walk(blob_root):
        for fn in files:
            if not fn.endswith(".safetensors"):
                continue
            tid = fn[: -len(".safetensors")]
            uploads.append((os.path.join(dirpath, fn),
                            f"{DEST_PREFIX}blobs/{tid[:2]}/{fn}"))
    uploads.append((meta_path, f"{DEST_PREFIX}master.db"))

    # Safety: every key must live under the destination prefix.
    for _p, key in uploads:
        assert key.startswith(DEST_PREFIX), f"refusing non-prefix key {key}"

    total_bytes = sum(os.path.getsize(p) for p, _ in uploads)
    print(f"uploading {len(uploads)} objects, {total_bytes / 1e9:.2f} GB "
          f"-> s3://{BUCKET}/{DEST_PREFIX}")

    s3 = boto3.client("s3", config=Config(
        region_name="us-east-1", max_pool_connections=args.workers + 4,
        retries={"mode": "standard", "max_attempts": 5}))

    done = {"n": 0, "bytes": 0}
    lock = threading.Lock()
    t0 = time.time()

    def put(item):
        path, key = item
        assert key.startswith(DEST_PREFIX)
        s3.upload_file(path, BUCKET, key)
        with lock:
            done["n"] += 1
            done["bytes"] += os.path.getsize(path)
            if done["n"] % 500 == 0 or done["n"] == len(uploads):
                dt = time.time() - t0
                print(f"  {done['n']}/{len(uploads)} "
                      f"({done['bytes'] / 1e9:.2f} GB, "
                      f"{done['bytes'] / 1e6 / dt:.0f} MB/s)", flush=True)

    with cf.ThreadPoolExecutor(args.workers) as pool:
        for f in cf.as_completed([pool.submit(put, u) for u in uploads]):
            f.result()

    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
