#!/usr/bin/env python3
"""Stage 0 S3 inventory (plan §3.3).

Lists s3://<bucket>/<root_prefix> without downloading objects, writes the
full key listing to eval/raw/s3_inventory.txt, and prints object count and
total bytes per first- and second-level prefix so the tensordex / zipnn /
raw layouts can be identified and filled into eval/config/object_layout.yaml.

Requires AWS credentials (instance profile or AWS_PROFILE); the bucket
denies anonymous access.

Usage:
    python eval/scripts/inventory_s3.py [--bucket tensor-tingfeng]
        [--prefix tensordb/] [--out eval/raw/s3_inventory.txt]
"""
from __future__ import annotations

import argparse
import collections
import os
import sys

import boto3
from botocore.config import Config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", default="tensor-tingfeng")
    ap.add_argument("--prefix", default="tensordb/")
    ap.add_argument("--out", default="eval/raw/s3_inventory.txt")
    ap.add_argument("--max-keys", type=int, default=None,
                    help="stop after this many keys (debugging)")
    args = ap.parse_args()

    s3 = boto3.client("s3", config=Config(
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        retries={"mode": "standard", "max_attempts": 5}))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    count = 0
    total = 0
    by_prefix: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    paginator = s3.get_paginator("list_objects_v2")
    with open(args.out, "w") as fh:
        for page in paginator.paginate(Bucket=args.bucket,
                                       Prefix=args.prefix):
            for obj in page.get("Contents", []):
                key, size = obj["Key"], obj["Size"]
                fh.write(f"{size}\t{key}\n")
                count += 1
                total += size
                rel = key[len(args.prefix):]
                parts = rel.split("/")
                for depth in (1, 2):
                    if len(parts) > depth:
                        p = "/".join(parts[:depth]) + "/"
                        by_prefix[p][0] += 1
                        by_prefix[p][1] += size
                if args.max_keys and count >= args.max_keys:
                    break
            else:
                continue
            break

    print(f"total: {count} objects, {total / 1e9:.2f} GB "
          f"under s3://{args.bucket}/{args.prefix}")
    print(f"{'prefix':50s} {'objects':>10s} {'GB':>10s}")
    for p in sorted(by_prefix):
        c, b = by_prefix[p]
        print(f"{p:50s} {c:10d} {b / 1e9:10.2f}")
    print(f"\nfull listing: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
