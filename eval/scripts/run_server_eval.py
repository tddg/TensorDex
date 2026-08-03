#!/usr/bin/env python3
"""Server-reconstruction eval orchestrator.

Per model: clear the reconstruction cache, then run the single-client
driver twice via --reps 2 — rep 0 is the cold-cache run (server pays
S3 closure fetch + decode), rep 1 is the warm-cache run (served from
disk). Cache is cleared between models so the disk never holds more
than one reconstructed model.

Prereqs: metadata_server and cache_server running (see eval/server/).

Usage:
    python eval/scripts/run_server_eval.py --models m1 m2 ... \
        --out eval/results/server_client
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request


def clear_cache(cache_url: str) -> None:
    req = urllib.request.Request(f"{cache_url}/admin/clear", method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        print(f"cache cleared: {r.read().decode().strip()}", flush=True)


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/healthz", timeout=5) as r:
            return json.loads(r.read()).get("ok") is True
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--metadata-url", default="http://127.0.0.1:8701")
    ap.add_argument("--cache-url", default="http://127.0.0.1:8702")
    ap.add_argument("--out", default="eval/results/server_client")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    for name, url in (("metadata", args.metadata_url),
                      ("cache", args.cache_url)):
        if not healthy(url):
            print(f"{name} server not healthy at {url}")
            return 1

    for model in args.models:
        clear_cache(args.cache_url)
        cmd = [sys.executable, "-u", "eval/scripts/run_single.py",
               "--models", model, "--systems", "tensordex_srv",
               "--reps", "2", "--seed", "42",
               "--workers", str(args.workers),
               "--metadata-url", args.metadata_url,
               "--cache-url", args.cache_url,
               "--out", args.out]
        print("== " + " ".join(cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"driver failed for {model} (rc={rc})")
            return rc
    clear_cache(args.cache_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
