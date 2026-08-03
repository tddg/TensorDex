"""Cache eviction daemon (plan Phase 2.4–2.5).

Watermark loop: when cache usage crosses the high watermark, evict
lowest-scoring entries until usage falls to the low watermark. Never
evicts pinned tids (in-flight materializations), ``.tmp`` litter is
ignored (cleaned when stale), and entries younger than --min-age are
exempt (they may be mid-first-serve).

Scoring is the research hook (§P2.5): ``score(tid) -> float`` — LOWER
is evicted first. Policies, A/B-comparable behind --policy:

  lru        score = atime (classic; relatime granularity accepted v1)
  dag_value  score = re-materialization cost this entry saves, summed
             over *resident* dependents (reverse-dependency index built
             from the manifest root), value-per-byte, atime tiebreak.
             A base whose resident children would each need it
             re-fetched + re-decoded scores far above a leaf nobody
             derives from.

Every eviction pass logs one JSONL line (policy, usage before/after,
evicted count/bytes, per-policy counters) — paper data, non-optional.

Usage mode is filesystem watermarks (--high/--low fractions of the
cache filesystem, the production config) or absolute directory bytes
(--high-bytes/--low-bytes, used by tests and shared-disk dev setups).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .cache import TensorCache


def build_reverse_deps(manifest_root: str):
    """tid -> {dependent tids one hop up} + tid -> re-mat cost estimate.

    Cost of re-materializing a delta if its base is gone ≈ its stored
    fetch bytes + decode work ∝ logical bytes. Summed over resident
    dependents at score time.
    """
    rdeps: dict[str, set] = {}
    cost: dict[str, float] = {}
    try:
        names = os.listdir(manifest_root)
    except FileNotFoundError:
        return rdeps, cost
    for name in names:
        if not name.endswith(".json") or name.endswith("@latest.json"):
            continue
        try:
            with open(os.path.join(manifest_root, name)) as fh:
                man = json.load(fh)
        except (OSError, ValueError):
            continue
        for c in man.get("closure", ()):
            cost[c["tid"]] = c["stored_bytes"] + 0.1 * c["logical_bytes"]
            if c["base_id"]:
                rdeps.setdefault(c["base_id"], set()).add(c["tid"])
    return rdeps, cost


def scores(entries, policy: str, rdeps, cost, resident: set):
    """{tid: score}; lower is evicted first."""
    if policy == "lru":
        return {tid: atime for tid, _, atime in entries}
    out = {}
    atimes = {tid: atime for tid, _, atime in entries}
    sizes = {tid: size for tid, size, _ in entries}
    for tid, size, atime in entries:
        saved = cost.get(tid, sizes.get(tid, 0))
        saved += sum(cost.get(d, 0)
                     for d in rdeps.get(tid, ()) if d in resident)
        # Value density + a tiny recency term so equal-value entries
        # still evict oldest-first.
        out[tid] = saved / max(size, 1) + 1e-12 * atime
    return out


def evict_pass(cache: TensorCache, *, policy: str, manifest_root: str,
               high_bytes: float, low_bytes: float, min_age: float,
               used_bytes: float, log) -> dict:
    if used_bytes < high_bytes:
        return {"evicted": 0, "evicted_bytes": 0, "used_bytes": used_bytes}
    entries = list(cache.entries())
    pinned = cache.pinned_tids()
    resident = {tid for tid, _, _ in entries}
    rdeps, cost = (build_reverse_deps(manifest_root)
                   if policy == "dag_value" else ({}, {}))
    sc = scores(entries, policy, rdeps, cost, resident)
    now = time.time()
    evicted = evicted_bytes = 0
    dir_used = cache.used_bytes()
    target_drop = used_bytes - low_bytes
    for tid, size, atime in sorted(entries, key=lambda e: sc[e[0]]):
        if evicted_bytes >= target_drop or dir_used <= 0:
            break
        if tid in pinned or now - atime < min_age:
            continue
        try:
            os.unlink(cache.path(tid))
        except OSError:
            continue
        evicted += 1
        evicted_bytes += size
        dir_used -= size
    rec = {"ts_ns": time.perf_counter_ns(), "event": "evict_pass",
           "policy": policy, "used_bytes": used_bytes,
           "high_bytes": high_bytes, "low_bytes": low_bytes,
           "candidates": len(entries), "pinned": len(pinned),
           "evicted": evicted, "evicted_bytes": evicted_bytes}
    log(rec)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="/mnt/nvme/cache")
    ap.add_argument("--manifest-root", default="/srv/manifests")
    ap.add_argument("--policy", choices=("lru", "dag_value"),
                    default="lru")
    ap.add_argument("--high", type=float, default=0.90,
                    help="filesystem-usage high watermark (fraction)")
    ap.add_argument("--low", type=float, default=0.80)
    ap.add_argument("--high-bytes", type=int, default=None,
                    help="absolute cache-dir watermarks (override)")
    ap.add_argument("--low-bytes", type=int, default=None)
    ap.add_argument("--min-age", type=float, default=60.0)
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--log", default="logs/evictor.jsonl")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)

    cache = TensorCache(args.cache_dir)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    fh = open(args.log, "a", buffering=1)

    def log(rec):
        fh.write(json.dumps(rec, separators=(",", ":")) + "\n")

    while True:
        if args.high_bytes is not None:
            used = cache.used_bytes()
            high, low = args.high_bytes, args.low_bytes
        else:
            used, total = cache.fs_usage()
            high, low = args.high * total, args.low * total
        evict_pass(cache, policy=args.policy,
                   manifest_root=args.manifest_root, high_bytes=high,
                   low_bytes=low, min_age=args.min_age, used_bytes=used,
                   log=log)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
