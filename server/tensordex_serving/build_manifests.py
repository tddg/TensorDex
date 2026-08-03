"""Offline manifest materialization job (plan Phase 1.1).

Reads the hub catalog and writes one immutable JSON manifest per model
version to the manifest root (EBS — durable), plus a `latest` symlink
per model. Runs at ingest/compaction time; the serving path never
computes manifests.

Usage:
    python -m tensordex_serving.build_manifests \
        --db eval/raw/hub_compressed/metadata.db \
        --out /srv/manifests --prefix compressed_eval
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from . import manifest as M


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True, help="manifest root dir")
    ap.add_argument("--prefix", default="compressed_eval",
                    help="S3 key prefix blobs live under")
    ap.add_argument("--model", action="append", default=None,
                    help="restrict to specific model(s); default: all")
    args = ap.parse_args(argv)

    con = M.connect_ro(args.db)
    try:
        models = args.model or M.list_models(con)
        for model in models:
            t0 = time.perf_counter()
            man = M.build_manifest(con, model, args.prefix)
            path = M.write_manifest(args.out, man)
            print(json.dumps({
                "model": model, "version": man["version"],
                "params": len(man["params"]),
                "closure": len(man["closure"]),
                "stored_bytes": sum(c["stored_bytes"]
                                    for c in man["closure"]),
                "logical_bytes": sum(p["logical_bytes"]
                                     for p in man["params"]),
                "build_ms": round((time.perf_counter() - t0) * 1e3, 2),
                "path": path}), flush=True)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
