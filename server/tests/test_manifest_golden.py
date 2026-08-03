"""Golden test: manifest builder vs eval adapter resolve (plan Phase 0).

The manifest builder must agree with
eval/src/adapters/tensordex.py::resolve on every model in the 9-model
compressed hub — same closure sets, base links, depths, stored/logical
sizes — including Woona's depth-2 chains.

Run:  .venv/bin/pytest server/tests/test_manifest_golden.py -q
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from eval.src.adapters.tensordex import TensorDexAdapter  # noqa: E402
from tensordex_serving import manifest as M  # noqa: E402

DB_DIR = os.path.join(REPO, "eval", "raw", "hub_compressed")
DB = os.path.join(DB_DIR, "metadata.db")
PREFIX = "compressed_eval"

pytestmark = pytest.mark.skipif(
    not os.path.exists(DB), reason="hub_compressed/metadata.db not present")


def adapter() -> TensorDexAdapter:
    # resolve() only touches the SQLite catalog — no S3 client needed.
    return TensorDexAdapter(s3_client=None, bucket="unused",
                            prefix=PREFIX, metadata_db_dir=DB_DIR)


def models() -> list[str]:
    con = M.connect_ro(DB)
    try:
        return M.list_models(con)
    finally:
        con.close()


@pytest.mark.parametrize("model", models())
def test_manifest_matches_adapter_resolve(model):
    plan = adapter().resolve(model)
    con = M.connect_ro(DB)
    try:
        man = M.build_manifest(con, model, PREFIX)
    finally:
        con.close()

    # Params: same name->tid mapping, logical sizes, dtypes, shapes.
    a_params = {t["tensor_name"]: t for t in plan.tensors}
    m_params = {p["name"]: p for p in man["params"]}
    assert set(a_params) == set(m_params)
    for name, ap in a_params.items():
        mp = m_params[name]
        assert mp["tid"] == ap["tid"]
        assert mp["logical_bytes"] == ap["logical_bytes"]
        assert mp["dtype"] == ap["dtype"]
        assert mp["shape"] == ap["shape"]

    # Closure: same blob set, keys, stored sizes, base links, depths.
    a_blobs = plan.extra["blobs"]
    a_depth = plan.extra["depth"]
    m_closure = {c["tid"]: c for c in man["closure"]}
    assert set(m_closure) == set(a_blobs)
    for tid, mc in m_closure.items():
        ab = a_blobs[tid]
        assert mc["stored_bytes"] == ab["stored"]
        assert mc["logical_bytes"] == ab["logical"]
        assert mc["base_id"] == ab["base_id"]
        assert mc["depth"] == a_depth[tid]
        assert mc["key"] == f"{PREFIX}/blobs/{tid[:2]}/{tid}.safetensors"
        # Every delta row carries its codec tag; direct blobs carry none.
        assert (mc["codec"] is not None) == (mc["base_id"] is not None)

    # Canonical: version is stable and content-derived.
    assert man["version"] == M.manifest_version(man)
    con = M.connect_ro(DB)
    try:
        again = M.build_manifest(con, model, PREFIX)
    finally:
        con.close()
    assert again["version"] == man["version"]


def test_woona_has_depth2_chains():
    con = M.connect_ro(DB)
    try:
        man = M.build_manifest(con, "AlexBefest/WoonaV1.2-9b", PREFIX)
    finally:
        con.close()
    assert max(c["depth"] for c in man["closure"]) == 2
