"""plan() correctness (plan Phase 1 test spec).

  have=∅                    ⇒ full closure
  have={bases}              ⇒ deltas only
  have={delta's parents}    ⇒ leaf only
  cross-model haves honored (shared tensors satisfy other manifests)
"""
from __future__ import annotations

import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

from tensordex_serving import manifest as M  # noqa: E402

DB = os.path.join(REPO, "eval", "raw", "hub_compressed", "metadata.db")
PREFIX = "compressed_eval"

pytestmark = pytest.mark.skipif(
    not os.path.exists(DB), reason="hub_compressed/metadata.db not present")


def build(model: str) -> dict:
    con = M.connect_ro(DB)
    try:
        return M.build_manifest(con, model, PREFIX)
    finally:
        con.close()


def deep_model() -> dict:
    """A manifest that actually has delta chains (depth >= 1)."""
    con = M.connect_ro(DB)
    try:
        for m in M.list_models(con):
            man = M.build_manifest(con, m, PREFIX)
            if any(c["depth"] for c in man["closure"]):
                return man
    finally:
        con.close()
    raise RuntimeError("no delta-compressed model in hub")


def test_empty_have_is_full_closure():
    man = deep_model()
    p = M.plan(man, set())
    assert {c["tid"] for c in p["fetch"]} == \
        {c["tid"] for c in man["closure"]}
    assert p["fetch_bytes"] == sum(
        c["stored_bytes"] for c in man["closure"])
    assert not p["satisfied"]


def test_have_bases_fetches_deltas_only():
    man = deep_model()
    bases = {c["tid"] for c in man["closure"] if c["base_id"] is None}
    p = M.plan(man, bases)
    fetched = {c["tid"] for c in p["fetch"]}
    # No fetched entry is a have...
    assert not (fetched & bases)
    # ...and everything else in the closure is still fetched.
    assert fetched == {c["tid"] for c in man["closure"]} - bases


def test_have_parent_chain_fetches_leaf_only():
    man = deep_model()
    deepest = max(man["closure"], key=lambda c: c["depth"])
    assert deepest["depth"] >= 1
    by_tid = {c["tid"]: c for c in man["closure"]}
    chain = []
    cur = deepest["base_id"]
    while cur:
        chain.append(cur)
        cur = by_tid[cur]["base_id"]
    p = M.plan(man, set(chain))
    needed_for_leaf = {c["tid"] for c in p["fetch"]
                       if c["tid"] == deepest["tid"]}
    assert needed_for_leaf == {deepest["tid"]}
    # The haves' subtrees are pruned: no ancestor of the leaf fetched.
    assert not ({c["tid"] for c in p["fetch"]} & set(chain))


def test_cross_model_haves_honored():
    """A tensor held from model A satisfies model B's manifest too
    (content addressing makes haves model-agnostic)."""
    con = M.connect_ro(DB)
    try:
        models = M.list_models(con)
        manifests = [M.build_manifest(con, m, PREFIX) for m in models]
    finally:
        con.close()
    for i, a in enumerate(manifests):
        a_tids = {c["tid"] for c in a["closure"]}
        for b in manifests[i + 1:]:
            shared = a_tids & {c["tid"] for c in b["closure"]}
            if shared:
                p = M.plan(b, shared)
                assert not ({c["tid"] for c in p["fetch"]} & shared)
                return
    pytest.skip("no shared tensors across eval models")
