"""Head-of-line byte-budget fetch/decode engine (plan Phase 0).

Ported from eval/src/adapters/tensordex.py::download — the engine that
saturated the NIC at 5.4 GB RSS in Stage 1 — decoupled from shard
writing and event recording so the materializer, the warmer, and future
clients can all drive it:

  - S3 fetches are submitted in DFS pre-order over the delta forest and
    their raw bytes count against an in-flight budget; the submitter
    blocks until decode consumes earlier blobs. Only the head-of-line
    blob (the one decode is blocked on) may exceed the budget — anything
    looser lets completed blobs accumulate without bound (the 3-OOM
    lesson).
  - Decoded bytes are kept only while a not-yet-decoded delta still
    references them (refcount); the live set is roughly one
    root-to-leaf path. Bases over the RAM cap spill to disk.
  - Every decoded tensor is content-hash verified against its tid
    before it is surfaced.

The caller supplies the closure entries (typically a manifest closure
or a /v1/plan fetch list), an ``on_tensor`` sink, and optionally a
``base_lookup`` for decoded bases that live outside this run (the NVMe
cache, a client base-cache) — entries pruned by plan() resolve their
bases through it.
"""
from __future__ import annotations

import concurrent.futures as cf
import os
import threading
import time

from .blob import parse_safetensors_bytes

CHUNK = 8 * 1024 * 1024


class BaseStore:
    """Decoded-base store with a RAM cap; overflow spills to disk."""

    def __init__(self, spill_dir: str, ram_budget: int = 4 * 1024 ** 3):
        self.spill_dir = spill_dir
        self.ram_budget = ram_budget
        self.ram: dict[str, bytes] = {}
        self.ram_bytes = 0
        self.on_disk: set[str] = set()

    def _path(self, tid: str) -> str:
        return os.path.join(self.spill_dir, tid)

    def put(self, tid: str, data: bytes):
        if self.ram_bytes + len(data) <= self.ram_budget:
            self.ram[tid] = data
            self.ram_bytes += len(data)
        else:
            os.makedirs(self.spill_dir, exist_ok=True)
            with open(self._path(tid), "wb") as fh:
                fh.write(data)
            self.on_disk.add(tid)

    def get(self, tid: str):
        if tid in self.ram:
            return self.ram[tid]
        if tid in self.on_disk:
            with open(self._path(tid), "rb") as fh:
                return fh.read()
        return None

    def drop(self, tid: str):
        data = self.ram.pop(tid, None)
        if data is not None:
            self.ram_bytes -= len(data)
        if tid in self.on_disk:
            self.on_disk.discard(tid)
            try:
                os.unlink(self._path(tid))
            except OSError:
                pass

    def clear(self):
        self.ram.clear()
        self.ram_bytes = 0
        for tid in list(self.on_disk):
            self.drop(tid)
        try:
            os.rmdir(self.spill_dir)
        except OSError:
            pass


def dfs_order(entries: list[dict]) -> tuple[list[dict], dict, dict]:
    """DFS pre-order over the delta forest of ``entries``.

    Returns (ordered entries, refcount by tid, entry by tid). A base is
    decoded immediately before its dependents so decoded bytes stay
    live only along the current root-to-leaf path. Entries whose base
    lies outside the set are roots here (their base resolves via
    base_lookup at decode time).
    """
    by_tid = {e["tid"]: e for e in entries}
    children: dict[str, list[str]] = {}
    roots = []
    for tid in sorted(by_tid):
        base = by_tid[tid]["base_id"]
        if base and base in by_tid:
            children.setdefault(base, []).append(tid)
        else:
            roots.append(tid)
    order: list[str] = []
    stack = list(reversed(roots))
    while stack:
        tid = stack.pop()
        order.append(tid)
        stack.extend(reversed(children.get(tid, ())))
    if len(order) != len(by_tid):
        raise ValueError("delta forest traversal incomplete")
    refcount = {t: len(children.get(t, ())) for t in order}
    return [by_tid[t] for t in order], refcount, by_tid


class ClosurePipeline:
    """Reusable fetch/decode engine over closure entries."""

    def __init__(self, s3_client, bucket: str, *, max_workers: int = 10,
                 inflight_budget: int = 1280 * 1024 ** 2,
                 base_ram_budget: int = 2560 * 1024 ** 2,
                 spill_dir: str = "/tmp/tensordex-basespill"):
        self.s3 = s3_client
        self.bucket = bucket
        self.max_workers = max_workers
        self.inflight_budget = inflight_budget
        self.base_ram_budget = base_ram_budget
        self.spill_dir = spill_dir

    def _fetch(self, entry: dict, pos: int, budget: dict) -> bytearray:
        size = entry["stored_bytes"]
        while True:
            with budget["lock"]:
                if (budget["bytes"] + size <= self.inflight_budget
                        or pos <= budget["head"]):
                    budget["bytes"] += size
                    break
            with budget["cv"]:
                budget["cv"].wait(0.05)
        resp = self.s3.get_object(Bucket=self.bucket, Key=entry["key"])
        # Preallocate at the exact object size (no bytearray-growth or
        # final bytes() copy — a 2 GB blob would transiently double).
        n = int(resp.get("ContentLength", size))
        buf = bytearray(n)
        view = memoryview(buf)
        off = 0
        for chunk in resp["Body"].iter_chunks(CHUNK):
            view[off:off + len(chunk)] = chunk
            off += len(chunk)
        if off != n:
            raise ValueError(f"short read for {entry['tid']}: {off} != {n}")
        return buf

    def run(self, entries: list[dict], on_tensor, *,
            base_lookup=None, spill_dir: str | None = None) -> dict:
        """Fetch + decode every entry; call on_tensor(entry, bytes) each.

        ``base_lookup(tid) -> bytes | None`` resolves decoded bases not
        present in ``entries`` (plan-pruned chains). Returns a stats
        dict (s3 gets/bytes, fetch/decode wall ms, tensors, bytes out).
        """
        from tensordex import _ops

        stats = {"s3_gets": 0, "s3_bytes": 0, "fetch_ms": 0.0,
                 "decode_ms": 0.0, "tensors": 0, "logical_bytes": 0}
        if not entries:
            return stats
        order, refcount, by_tid = dfs_order(entries)
        decoded = BaseStore(spill_dir or self.spill_dir,
                            ram_budget=self.base_ram_budget)
        budget = {"bytes": 0, "head": 0, "lock": threading.Lock(),
                  "cv": threading.Condition()}
        # External bases (outside `entries`) are fetched via base_lookup
        # once and refcounted across the deltas that need them.
        ext_refs: dict[str, int] = {}
        for e in order:
            b = e["base_id"]
            if b and b not in by_tid:
                ext_refs[b] = ext_refs.get(b, 0) + 1

        try:
            with cf.ThreadPoolExecutor(self.max_workers) as pool:
                futs = {e["tid"]: pool.submit(self._fetch, e, pos, budget)
                        for pos, e in enumerate(order)}
                for pos, entry in enumerate(order):
                    tid = entry["tid"]
                    with budget["lock"]:
                        budget["head"] = pos
                    with budget["cv"]:
                        budget["cv"].notify_all()
                    t0 = time.perf_counter_ns()
                    raw = futs.pop(tid).result()
                    stats["fetch_ms"] += (time.perf_counter_ns() - t0) / 1e6
                    stats["s3_gets"] += 1
                    stats["s3_bytes"] += len(raw)
                    meta, _k, (_dt, _sh, payload) = \
                        parse_safetensors_bytes(raw)
                    base_id = meta.get("base_tensor_id") or entry["base_id"]
                    t0 = time.perf_counter_ns()
                    if not base_id:
                        out = bytes(payload)
                    else:
                        base_bytes = decoded.get(base_id)
                        if base_bytes is None and base_lookup is not None:
                            base_bytes = base_lookup(base_id)
                            # Keep an external base resident while later
                            # entries still reference it.
                            if base_bytes is not None and \
                                    ext_refs.get(base_id, 0) > 1:
                                decoded.put(base_id, base_bytes)
                                refcount[base_id] = ext_refs[base_id]
                        if base_bytes is None:
                            raise ValueError(f"base {base_id} unavailable "
                                             f"for delta {tid}")
                        item_size = int(meta.get("item_size", 2))
                        out = bytes(_ops.decompress_tensorx_rust(
                            bytes(payload), base_bytes, item_size))
                        base_bytes = None
                    stats["decode_ms"] += (time.perf_counter_ns() - t0) / 1e6
                    del raw, payload
                    with budget["lock"]:
                        budget["bytes"] -= entry["stored_bytes"]
                    with budget["cv"]:
                        budget["cv"].notify_all()

                    got = _ops.content_hash(out)
                    if got != tid:
                        raise ValueError(f"hash mismatch: expected {tid}, "
                                         f"got {got}")
                    on_tensor(entry, out)
                    stats["tensors"] += 1
                    stats["logical_bytes"] += len(out)
                    if refcount.get(tid, 0) > 0:
                        decoded.put(tid, out)
                    if base_id and base_id in refcount:
                        refcount[base_id] -= 1
                        if refcount[base_id] == 0:
                            decoded.drop(base_id)
                    out = None  # release before next iteration
        finally:
            decoded.clear()
        return stats
