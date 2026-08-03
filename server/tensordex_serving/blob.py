"""Safetensors blob parsing for TensorDex hub blobs.

A TensorDex blob holds ``_fingerprint``, the tensor under its own name,
and ``<name>_fingerprint`` (see ae/_blobs.py); only the real tensor key
matters. Per-tensor codec metadata is a JSON *string* keyed by the
tensor name inside ``__metadata__`` (e.g. ``{"tensor": "{\"codec\":
\"tensorx\", \"item_size\": 2}"}``) — ``item_size`` is required to
decode f32 deltas correctly.

Ported verbatim from eval/src/adapters/tensordex.py (proven on all 54
Stage 1 runs); kept dependency-free.
"""
from __future__ import annotations

import json
import struct

# torch dtype -> (safetensors dtype, item size)
DTYPE_MAP = {
    "torch.bfloat16": ("BF16", 2), "torch.float16": ("F16", 2),
    "torch.float32": ("F32", 4), "torch.float64": ("F64", 8),
    "torch.int64": ("I64", 8), "torch.int32": ("I32", 4),
    "torch.int16": ("I16", 2), "torch.int8": ("I8", 1),
    "torch.uint8": ("U8", 1), "torch.bool": ("BOOL", 1),
}


def logical_nbytes(dtype: str, shape: list[int]) -> int:
    """Logical (decoded) byte size from torch dtype + shape.

    The hub's tensors.size_bytes is the *stored* (post-compress) size;
    logical sizes must always be recomputed from shape x dtype.
    """
    nel = 1
    for d in shape:
        nel *= d
    return nel * DTYPE_MAP[dtype][1]


def parse_safetensors_bytes(data):
    """Return (metadata_dict, real_key, (dtype, shape, payload_view)).

    The payload is a memoryview into ``data`` (no copy).
    """
    (hlen,) = struct.unpack("<Q", bytes(data[:8]))
    header = json.loads(bytes(data[8:8 + hlen]))
    meta = header.pop("__metadata__", {}) or {}
    body = memoryview(data)[8 + hlen:]
    for name, info in header.items():
        if name == "_fingerprint" or name.endswith("_fingerprint"):
            continue
        tmeta = meta.get(name, {})
        if isinstance(tmeta, str):
            tmeta = json.loads(tmeta)
        b0, b1 = info["data_offsets"]
        return tmeta, name, (info["dtype"], info["shape"], body[b0:b1])
    raise ValueError("no real tensor key in blob header")
