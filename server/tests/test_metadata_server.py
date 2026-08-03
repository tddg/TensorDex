"""Metadata server API tests (plan Phase 1.4) — TestClient, no network.

Covers: manifest ETag/304/immutable headers, prebuilt-vs-SQLite-fallback
equivalence, closure chain ordering, /v1/plan negotiation, model listing
with sizes, health/metrics.
"""
from __future__ import annotations

import importlib
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

DB = os.path.join(REPO, "eval", "raw", "hub_compressed", "metadata.db")

pytestmark = pytest.mark.skipif(
    not os.path.exists(DB), reason="hub_compressed/metadata.db not present")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("manifests")
    os.environ.update(
        TDS_DB=DB, TDS_MANIFEST_ROOT=str(root), TDS_PREFIX="compressed_eval",
        TDS_LOG=str(tmp_path_factory.mktemp("logs") / "md.jsonl"))
    import tensordex_serving.metadata_server as ms
    importlib.reload(ms)
    from fastapi.testclient import TestClient

    # Materialize manifests for half the models so both the prebuilt
    # path and the SQLite fallback are exercised.
    from tensordex_serving import build_manifests, manifest as M
    con = M.connect_ro(DB)
    models = M.list_models(con)
    con.close()
    build_manifests.main(["--db", DB, "--out", str(root),
                          "--prefix", "compressed_eval"]
                         + sum((["--model", m]
                                for m in models[:len(models) // 2]), []))
    return TestClient(ms.app), models, str(root)


def test_healthz(client):
    c, _, _ = client
    assert c.get("/healthz").json() == {"ok": True}


def test_models_listing(client):
    c, models, _ = client
    got = c.get("/v1/models").json()["models"]
    assert {m["model"] for m in got} == set(models)
    prebuilt = [m for m in got if m["version"]]
    assert prebuilt, "expected some materialized manifests"
    for m in prebuilt:
        assert m["logical_bytes"] > 0
        assert 0 < m["stored_bytes"] < m["logical_bytes"] * 2


def test_manifest_etag_and_fallback(client):
    c, models, root = client
    for model in (models[0], models[-1]):  # prebuilt + fallback
        r = c.get("/v1/manifest", params={"model": model})
        assert r.status_code == 200
        man = r.json()
        assert man["model"] == model
        assert r.headers["etag"] == man["version"]
        assert "immutable" in r.headers["cache-control"]
        r304 = c.get("/v1/manifest", params={"model": model},
                     headers={"If-None-Match": man["version"]})
        assert r304.status_code == 304


def test_manifest_fallback_equals_prebuilt(client):
    c, models, root = client
    # The last model has no materialized file — fallback path.
    fallback = c.get("/v1/manifest",
                     params={"model": models[-1]}).json()
    from tensordex_serving import manifest as M
    con = M.connect_ro(DB)
    direct = M.build_manifest(con, models[-1], "compressed_eval")
    con.close()
    assert fallback == direct


def test_manifest_unknown_404(client):
    c, _, _ = client
    assert c.get("/v1/manifest",
                 params={"model": "no/such-model"}).status_code == 404


def test_closure_chain_order(client):
    c, models, _ = client
    man = c.get("/v1/manifest", params={"model": models[0]}).json()
    deep = max(man["closure"], key=lambda x: x["depth"])
    r = c.get("/v1/closure", params={"tid": deep["tid"]})
    chain = r.json()["chain"]
    assert chain[0]["tid"] == deep["tid"]
    assert len(chain) == deep["depth"] + 1
    assert chain[-1]["base_id"] is None  # root last
    for above, below in zip(chain, chain[1:]):
        assert above["base_id"] == below["tid"]
    assert c.get("/v1/closure",
                 params={"tid": "0" * 32}).status_code == 404


def test_plan_endpoint(client):
    c, models, _ = client
    man = c.get("/v1/manifest", params={"model": models[0]}).json()
    full = c.post("/v1/plan", json={"model": models[0], "have": []}).json()
    assert {f["tid"] for f in full["fetch"]} == \
        {x["tid"] for x in man["closure"]}
    bases = [x["tid"] for x in man["closure"] if x["base_id"] is None]
    part = c.post("/v1/plan",
                  json={"model": models[0], "have": bases}).json()
    assert not ({f["tid"] for f in part["fetch"]} & set(bases))
    assert part["fetch_bytes"] < full["fetch_bytes"]
    assert c.post("/v1/plan", json={"model": "no/such", "have": []}
                  ).status_code == 404


def test_metrics(client):
    c, _, _ = client
    text = c.get("/metrics").text
    assert "tds_requests_total" in text
    assert 'path="/v1/manifest"' in text
