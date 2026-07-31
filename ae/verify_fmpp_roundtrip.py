#!/usr/bin/env python3
"""Self-contained byte-exact FM++ round-trip verification.

By default this generates deterministic pairs of synthetic 2-byte tensor
elements, then executes the complete codec path:

    target + base -> FM++ delta -> target

No artifact cache, model download, or tensor store is required.
"""
from __future__ import annotations

import argparse
import os
import random
import sys

_AE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _AE_DIR)

from _blobs import (  # noqa: E402
    HAS_FMPP,
    fmpp_roundtrip,
)


def synthetic_pairs(n: int, pair_kib: int, seed: int):
    """Generate deterministic base/target byte pairs resembling 2-byte weights."""
    pair_bytes = pair_kib * 1024
    if pair_bytes <= 0 or pair_bytes % 2:
        raise ValueError("--pair-kib must produce a positive, even byte length")

    # Build a small distinct word pattern per pair, then repeat it.  This keeps
    # generation fast while exercising nonzero signed residuals throughout the
    # stream instead of testing only identical buffers.
    pattern_bytes = min(pair_bytes, 64 * 1024)
    for pair_index in range(n):
        rng = random.Random(seed + pair_index)
        base_pattern = bytearray(pattern_bytes)
        target_pattern = bytearray(pattern_bytes)
        for offset in range(0, pattern_bytes, 2):
            base_word = rng.getrandbits(16)
            delta = ((offset // 2 + pair_index) % 65) - 32
            target_word = (base_word + delta) & 0xFFFF
            base_pattern[offset:offset + 2] = base_word.to_bytes(2, "little")
            target_pattern[offset:offset + 2] = target_word.to_bytes(2, "little")
        repeats = (pair_bytes + pattern_bytes - 1) // pattern_bytes
        base = (bytes(base_pattern) * repeats)[:pair_bytes]
        target = (bytes(target_pattern) * repeats)[:pair_bytes]
        yield f"synthetic-{pair_index:04d}", target, base


def main() -> int:
    ap = argparse.ArgumentParser(
        description="TensorDex AE - FM++ byte-exact round-trip verification"
    )
    ap.add_argument("--n", type=int, default=16, help="number of pairs (default: 16)")
    ap.add_argument("--pair-kib", type=int, default=1024,
                    help="synthetic bytes per pair in KiB (default: 1024)")
    ap.add_argument("--seed", type=int, default=0, help="generation seed")
    args = ap.parse_args()

    if not HAS_FMPP:
        print("ERROR: FM++ encoder/decoder not available; run `make ae-fmpp` first.")
        return 2
    if args.n <= 0:
        print("ERROR: --n must be positive")
        return 2

    try:
        pairs = list(synthetic_pairs(args.n, args.pair_kib, args.seed))
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 2
    if not pairs:
        print("ERROR: no eligible FM++ pairs found")
        return 2

    print(
        f"FM++ round trip: {len(pairs)} pairs from deterministic synthetic data "
        f"(seed={args.seed})"
    )
    passed = 0
    total_raw = 0
    total_compressed = 0
    for i, (label, target, base) in enumerate(pairs, 1):
        compressed_bytes, exact = fmpp_roundtrip(target, base)
        passed += exact
        total_raw += len(target)
        total_compressed += compressed_bytes
        if i <= 10 or not exact:
            print(
                f"  [{i:4}/{len(pairs)}] {'OK  ' if exact else 'FAIL'} "
                f"{label}  {compressed_bytes}/{len(target)} bytes"
            )

    if len(pairs) > 10:
        print(f"  ... ({len(pairs) - 10} more)")
    print(f"\ncompressed/raw: {total_compressed / total_raw:.3f}x")
    print(f"FM++ round trip: {passed}/{len(pairs)} byte-exact")
    if passed != len(pairs):
        print("RESULT: FAIL - decoded bytes differ from the original target")
        return 1
    print("RESULT: PASS - every decoded tensor is byte-for-byte identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
