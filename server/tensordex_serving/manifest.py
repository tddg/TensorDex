"""Immutable per-model manifests over a hub metadata.db (plan Phase 0).

Manifest schema (one JSON document per model version):

    {"model":   "<model_name>",
     "version": "<sha256 of the canonicalized manifest body>",
     "params":  [{"name", "tid", "dtype", "shape", "logical_bytes"}],
     "closure": [{"tid", "key", "stored_bytes", "logical_bytes",
                  "codec", "base_id", "depth"}]}

``params`` maps every parameter to its tensor id; ``closure`` is the
full delta-graph closure — every blob a cold client must fetch. Both
lists are sorted (params by name, closure by tid) so the document is
canonical; ``version`` is the content hash of the body, making the
manifest self-verifying and safely cacheable as immutable.

The builder must agree with eval/src/adapters/tensordex.py::resolve on
all 9 eval models (golden test) — same closure sets, base links, depths
and sizes, including Woona's depth-2 chains.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3

from .blob import logical_nbytes

SQL_CHUNK = 500


def connect_ro(db_path: str) -> sqlite3.Connection:
    """Read-only URI connection (WAL-friendly, never creates the file)."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True,
                           check_same_thread=False)


def blob_key(prefix: str, tid: str) -> str:
    return f"{prefix.rstrip('/')}/blobs/{tid[:2]}/{tid}.safetensors"


def model_slug(model: str) -> str:
    """Filesystem-safe manifest filename component."""
    return model.replace("/", "_")


def list_models(con: sqlite3.Connection) -> list[str]:
    return [r[0] for r in con.execute(
        "SELECT DISTINCT model_name FROM model_mappings ORDER BY 1")]


def _has_deltas(con: sqlite3.Connection) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='tensor_deltas'").fetchone() is not None


def build_manifest(con: sqlite3.Connection, model: str,
                   key_prefix: str) -> dict:
    """Build the manifest for one model from the catalog.

    Iterative closure over the delta graph, chunked IN queries (the
    live catalog has 3.4k+ tensors per model). Raises KeyError for an
    unknown model, ValueError on a broken graph (missing tensor rows or
    a delta cycle).
    """
    mapping = con.execute(
        "SELECT param_name, tensor_id FROM model_mappings "
        "WHERE model_name = ?", (model,)).fetchall()
    if not mapping:
        raise KeyError(model)
    deltas_on = _has_deltas(con)

    blobs: dict[str, dict] = {}
    frontier = sorted({t for _, t in mapping})
    while frontier:
        nxt: set[str] = set()
        for i in range(0, len(frontier), SQL_CHUNK):
            chunk = frontier[i:i + SQL_CHUNK]
            ph = ",".join("?" * len(chunk))
            rows = con.execute(
                f"SELECT id, shape, dtype, size_bytes FROM tensors "
                f"WHERE id IN ({ph})", chunk).fetchall()
            if len(rows) != len(chunk):
                missing = sorted(set(chunk) - {r[0] for r in rows})
                raise ValueError(f"tensors missing from metadata: "
                                 f"{missing[:5]}...")
            dmap = {r[0]: (r[1], r[2]) for r in con.execute(
                f"SELECT tensor_id, base_tensor_id, codec FROM "
                f"tensor_deltas WHERE tensor_id IN ({ph})",
                chunk)} if deltas_on else {}
            for tid, shape_json, dtype, stored in rows:
                shape = json.loads(shape_json)
                base, codec = dmap.get(tid, (None, None))
                blobs[tid] = {
                    "tid": tid, "key": blob_key(key_prefix, tid),
                    "stored_bytes": stored,
                    "logical_bytes": logical_nbytes(dtype, shape),
                    "codec": codec, "base_id": base,
                    "dtype": dtype, "shape": shape}
                if base and base not in blobs:
                    nxt.add(base)
        frontier = sorted(nxt - set(blobs))

    # Dependency depth per tid (0 = direct storage), cycle-checked.
    depth: dict[str, int] = {}

    def d(tid: str, seen=()) -> int:
        if tid in depth:
            return depth[tid]
        if tid in seen:
            raise ValueError(f"delta cycle at {tid}")
        b = blobs[tid]["base_id"]
        depth[tid] = 0 if not b else 1 + d(b, seen + (tid,))
        return depth[tid]

    for t in blobs:
        d(t)

    params = [{"name": p, "tid": t, "dtype": blobs[t]["dtype"],
               "shape": blobs[t]["shape"],
               "logical_bytes": blobs[t]["logical_bytes"]}
              for p, t in sorted(mapping)]
    closure = [{"tid": b["tid"], "key": b["key"],
                "stored_bytes": b["stored_bytes"],
                "logical_bytes": b["logical_bytes"],
                "codec": b["codec"], "base_id": b["base_id"],
                "depth": depth[b["tid"]]}
               for b in sorted(blobs.values(), key=lambda b: b["tid"])]

    body = {"model": model, "params": params, "closure": closure}
    return {"model": model, "version": manifest_version(body),
            "params": params, "closure": closure}


def manifest_version(body: dict) -> str:
    """Content hash of the canonicalized manifest body (sans version)."""
    canon = json.dumps(
        {"model": body["model"], "params": body["params"],
         "closure": body["closure"]},
        separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(canon).hexdigest()


def manifest_path(root: str, model: str, version: str) -> str:
    return os.path.join(root, f"{model_slug(model)}@{version}.json")


def latest_path(root: str, model: str) -> str:
    return os.path.join(root, f"{model_slug(model)}@latest.json")


def write_manifest(root: str, man: dict) -> str:
    """Materialize one manifest + refresh the model's `latest` symlink.

    Atomic tmp+rename; the versioned file is immutable so a re-run with
    unchanged content is a no-op rename over an identical file.
    """
    os.makedirs(root, exist_ok=True)
    path = manifest_path(root, man["model"], man["version"])
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(man, fh, separators=(",", ":"))
    os.replace(tmp, path)
    link = latest_path(root, man["model"])
    tmp_link = f"{link}.tmp.{os.getpid()}"
    try:
        os.unlink(tmp_link)
    except OSError:
        pass
    os.symlink(os.path.basename(path), tmp_link)
    os.replace(tmp_link, link)
    return path


def load_manifest(root: str, model: str, version: str | None = None):
    """Return the stored manifest dict, or None if not materialized."""
    path = (manifest_path(root, model, version) if version
            else latest_path(root, model))
    try:
        with open(path) as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


# ---- /v1/plan negotiation (research hook, §P1.4) ---------------------

def plan(man: dict, have: set[str]) -> dict:
    """State-negotiated transfer plan over one manifest.

    v1 semantics: a `have` satisfies its entire subtree requirement —
    the client holds *decoded* bytes for that tid, so nothing below it
    in the chain is needed for the params it feeds. Returns the closure
    entries still to fetch, in the manifest's canonical order. Pure
    function; no DB access.
    """
    by_tid = {c["tid"]: c for c in man["closure"]}
    needed: set[str] = set()
    for p in man["params"]:
        cur = p["tid"]
        while cur and cur not in have:
            if cur in needed:
                break  # chain suffix already walked via another param
            needed.add(cur)
            cur = by_tid[cur]["base_id"]
    fetch = [c for c in man["closure"] if c["tid"] in needed]
    return {"model": man["model"], "version": man["version"],
            "fetch": fetch,
            "fetch_bytes": sum(c["stored_bytes"] for c in fetch),
            "satisfied": sorted(have & set(by_tid)),
            "reconstructed_fallback": False}
