"""Shared library for the TensorDex hub serving tier (plan Phase 0).

Importable by the metadata server, the materializer, the evictor, and
tests. Three modules:

    manifest  — immutable per-model manifest schema + builder over a hub
                metadata.db (golden-tested against the eval adapter's
                resolve()).
    blob      — safetensors blob parsing, incl. the nested
                ``__metadata__[name]`` JSON-string codec metadata.
    pipeline  — the head-of-line byte-budget fetch/decode engine ported
                from eval/src/adapters/tensordex.py (NIC-saturating at
                bounded RSS).
"""

from . import blob, manifest, pipeline  # noqa: F401

__all__ = ["blob", "manifest", "pipeline"]
