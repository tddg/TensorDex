"""Hot-cache tests (plan Phase 2.7) — synthetic blobs, in-proc fake S3.

Covers: cold/warm correctness (hash at client), delta-chain decode,
concurrent-miss coalescing, warm endpoint, empty-cache restart,
eviction under synthetic fill (both policies), pin protection.

The Woona-under-8GB RAM ceiling and real-S3 byte-exactness run in the
smoke test (real bucket), not here.
"""
from __future__ import annotations

import importlib
import json
import os
import struct
import sys
import threading
import time

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from tensordex import _ops  # noqa: E402

PREFIX = "test_prefix"
ITEM = 4  # f32


def key_of(tid: str) -> str:
    return f"{PREFIX}/blobs/{tid[:2]}/{tid}.safetensors"


def make_blob(name: str, payload: bytes, n_el: int, meta: dict) -> bytes:
    header = {
        name: {"dtype": "F32", "shape": [n_el],
               "data_offsets": [0, len(payload)]},
        "__metadata__": {name: json.dumps(meta)},
    }
    hjson = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(hjson)) + hjson + payload


class FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def iter_chunks(self, chunk):
        for i in range(0, len(self.data), chunk):
            yield self.data[i:i + chunk]


class FakeS3:
    def __init__(self, objects: dict):
        self.objects = objects
        self.gets = 0
        self.lock = threading.Lock()

    def get_object(self, Bucket, Key):  # noqa: N803
        with self.lock:
            self.gets += 1
        data = self.objects[Key]
        return {"ContentLength": len(data), "Body": FakeBody(data)}


def synth_chain(depth: int, n_el: int = 4096):
    """Root -> delta -> delta... Returns (closure chain tid-first,
    fake-S3 objects, expected raw bytes by tid)."""
    torch.manual_seed(depth * 1000 + n_el)
    raws = [torch.randn(n_el).float().numpy().tobytes()]
    for _ in range(depth):
        prev = torch.frombuffer(bytearray(raws[-1]), dtype=torch.float32)
        nxt = (prev + torch.randn(n_el) * 1e-3).numpy().tobytes()
        raws.append(nxt)
    tids = [_ops.content_hash(r) for r in raws]
    objects, chain, expect = {}, [], {}
    for i, raw in enumerate(raws):
        tid = tids[i]
        expect[tid] = raw
        if i == 0:
            payload, meta = raw, {}
            stored = len(raw)
        else:
            payload = bytes(_ops.compress_tensorx_rust(
                raw, raws[i - 1], ITEM, 3))
            meta = {"codec": "tensorx", "item_size": ITEM,
                    "base_tensor_id": tids[i - 1]}
            stored = len(payload)
        objects[key_of(tid)] = make_blob("t", payload, n_el, meta)
        chain.append({"tid": tid, "key": key_of(tid),
                      "stored_bytes": stored, "logical_bytes": len(raw),
                      "codec": meta.get("codec"),
                      "base_id": tids[i - 1] if i else None,
                      "depth": i})
    chain.reverse()  # tid first (deepest), root last
    return chain, objects, expect


@pytest.fixture()
def mat(tmp_path, monkeypatch):
    monkeypatch.setenv("TDM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("TDM_SPILL_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("TDM_LOG", str(tmp_path / "logs" / "m.jsonl"))
    import tensordex_serving.materializer as m
    importlib.reload(m)
    return m


def client_of(m):
    from fastapi.testclient import TestClient
    return TestClient(m.app)


def wire(m, objects, chains):
    """Attach fake S3 + closure/plan fetchers for synthetic tids."""
    fake = FakeS3(objects)
    m._S3 = fake
    by_tid = {}
    for chain in chains:
        for i, ent in enumerate(chain):
            by_tid[ent["tid"]] = chain[i:]
    m.fetch_closure = lambda tid: by_tid[tid]
    return fake


def test_cold_warm_and_restart(mat):
    chain, objects, expect = synth_chain(depth=0)
    fake = wire(mat, objects, [chain])
    c = client_of(mat)
    tid = chain[0]["tid"]

    r = c.get(f"/v1/tensor/{tid}")
    assert r.status_code == 200
    assert r.headers["x-cache"] == "miss"
    assert _ops.content_hash(r.content) == tid  # hash at client
    assert r.content == expect[tid]

    r2 = c.get(f"/v1/tensor/{tid}")
    assert r2.headers["x-cache"] == "hit"
    assert fake.gets == 1  # warm hit did not refetch

    # Empty-cache restart: wipe everything, same request works again.
    import shutil
    shutil.rmtree(mat.cache.root)
    mat.cache.__init__(mat.cache.root)
    r3 = c.get(f"/v1/tensor/{tid}")
    assert r3.headers["x-cache"] == "miss"
    assert r3.content == expect[tid]


def test_delta_chain_decode_and_ancestor_reuse(mat):
    chain, objects, expect = synth_chain(depth=2)
    fake = wire(mat, objects, [chain])
    c = client_of(mat)
    leaf, mid, root = chain[0], chain[1], chain[2]

    r = c.get(f"/v1/tensor/{leaf['tid']}")
    assert r.content == expect[leaf["tid"]]
    assert fake.gets == 3  # whole chain fetched
    # Every chain member was cached on the way up.
    for ent in chain:
        assert mat.cache.has(ent["tid"])

    # A sibling delta off `mid` reuses the cached ancestor: only the
    # new delta blob is fetched.
    torch.manual_seed(7)
    n_el = 4096
    mid_raw = expect[mid["tid"]]
    sib_raw = (torch.frombuffer(bytearray(mid_raw), dtype=torch.float32)
               + torch.randn(n_el) * 1e-3).numpy().tobytes()
    sib_tid = _ops.content_hash(sib_raw)
    payload = bytes(_ops.compress_tensorx_rust(sib_raw, mid_raw, ITEM, 3))
    objects[key_of(sib_tid)] = make_blob(
        "t", payload, n_el, {"codec": "tensorx", "item_size": ITEM,
                             "base_tensor_id": mid["tid"]})
    sib_ent = {"tid": sib_tid, "key": key_of(sib_tid),
               "stored_bytes": len(payload), "logical_bytes": len(sib_raw),
               "codec": "tensorx", "base_id": mid["tid"], "depth": 2}
    old_fetch = mat.fetch_closure
    mat.fetch_closure = lambda tid: ([sib_ent] + old_fetch(mid["tid"])
                                     if tid == sib_tid else old_fetch(tid))
    before = fake.gets
    r = c.get(f"/v1/tensor/{sib_tid}")
    assert r.content == sib_raw
    assert fake.gets == before + 1  # ancestor came from cache, not S3


def test_concurrent_miss_coalescing(mat):
    chain, objects, expect = synth_chain(depth=1)
    fake = wire(mat, objects, [chain])
    c = client_of(mat)
    tid = chain[0]["tid"]
    results = []

    def hit():
        results.append(c.get(f"/v1/tensor/{tid}").content)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == expect[tid] for r in results)
    assert fake.gets == 2  # one chain fetch total, not 8


def test_accel_redirect_mode(mat):
    chain, objects, _ = synth_chain(depth=0)
    wire(mat, objects, [chain])
    c = client_of(mat)
    tid = chain[0]["tid"]
    r = c.get(f"/v1/tensor/{tid}", headers={"X-Accel-Expected": "1"})
    assert r.status_code == 200
    assert r.headers["x-accel-redirect"] == f"/_cache/{tid[:2]}/{tid}"
    assert not r.content  # nginx does the send


def test_warm_endpoint(mat):
    chain, objects, expect = synth_chain(depth=2)
    wire(mat, objects, [chain])
    model = "synthetic/model"
    mat.fetch_plan = lambda m, have: {
        "model": m, "version": "v",
        "fetch": [e for e in chain if e["tid"] not in have],
        "fetch_bytes": sum(e["stored_bytes"] for e in chain)}
    c = client_of(mat)
    r = c.post("/v1/warm", json={"model": model})
    assert r.json()["state"] == "running"
    for _ in range(100):
        s = c.get(f"/v1/warm/{model.replace('/', '_')}").json()
        if s["state"] != "running":
            break
        time.sleep(0.05)
    assert s["state"] == "done"
    assert s["stats"]["tensors"] == 3
    for ent in chain:
        assert mat.cache.get_bytes(ent["tid"]) == expect[ent["tid"]]


# ---- eviction --------------------------------------------------------

def fill(cache, spec):
    """spec: [(tid, size, age_s)] — synthesize entries w/ back-dated
    atimes (no hash verify)."""
    now = time.time()
    for tid, size, age in spec:
        cache.put(tid, b"x" * size, verify=False)
        os.utime(cache.path(tid), (now - age, now - age))


def test_eviction_lru_pins_and_min_age(tmp_path):
    from tensordex_serving.cache import TensorCache
    from tensordex_serving.evictor import evict_pass
    cache = TensorCache(str(tmp_path / "c"))
    fill(cache, [("aa" + "0" * 30, 1000, 500),   # oldest -> evicted
                 ("bb" + "0" * 30, 1000, 400),   # next    -> evicted
                 ("cc" + "0" * 30, 1000, 300),   # pinned  -> kept
                 ("dd" + "0" * 30, 1000, 30),    # < min-age -> kept
                 ("ee" + "0" * 30, 1000, 10)])   # < min-age -> kept
    logs = []
    with cache.pinned(["cc" + "0" * 30]):
        rec = evict_pass(cache, policy="lru", manifest_root="/nope",
                         high_bytes=4000, low_bytes=2500, min_age=60,
                         used_bytes=5000, log=logs.append)
    assert rec["evicted"] == 2
    assert not cache.has("aa" + "0" * 30)
    assert not cache.has("bb" + "0" * 30)
    assert cache.has("cc" + "0" * 30)  # pin respected
    assert cache.has("ee" + "0" * 30)  # min-age respected
    # Below high watermark: no-op pass.
    rec2 = evict_pass(cache, policy="lru", manifest_root="/nope",
                      high_bytes=4000, low_bytes=2500, min_age=60,
                      used_bytes=3000, log=logs.append)
    assert rec2["evicted"] == 0


def test_eviction_dag_value_keeps_shared_base(tmp_path):
    from tensordex_serving.cache import TensorCache
    from tensordex_serving.evictor import evict_pass
    cache = TensorCache(str(tmp_path / "c"))
    base, d1, d2, loner = ("ba" + "0" * 30, "d1" + "0" * 30,
                           "d2" + "0" * 30, "f0" + "0" * 30)
    # Same size + same atime: only the DAG can break the tie.
    fill(cache, [(t, 1000, 300) for t in (base, d1, d2, loner)])
    root = tmp_path / "manifests"
    root.mkdir()
    man = {"model": "m", "version": "v", "params": [],
           "closure": [
               {"tid": base, "base_id": None, "stored_bytes": 1000,
                "logical_bytes": 1000, "codec": None, "depth": 0,
                "key": "k"},
               {"tid": d1, "base_id": base, "stored_bytes": 1000,
                "logical_bytes": 1000, "codec": "tensorx", "depth": 1,
                "key": "k"},
               {"tid": d2, "base_id": base, "stored_bytes": 1000,
                "logical_bytes": 1000, "codec": "tensorx", "depth": 1,
                "key": "k"},
               {"tid": loner, "base_id": None, "stored_bytes": 1000,
                "logical_bytes": 1000, "codec": None, "depth": 0,
                "key": "k"}]}
    (root / "m@v.json").write_text(json.dumps(man))
    rec = evict_pass(cache, policy="dag_value",
                     manifest_root=str(root), high_bytes=3500,
                     low_bytes=3000, min_age=60, used_bytes=4000,
                     log=lambda r: None)
    assert rec["evicted"] == 1
    assert cache.has(base)      # two resident dependents -> highest value
    assert not cache.has(loner)  # nothing derives from it -> evicted
