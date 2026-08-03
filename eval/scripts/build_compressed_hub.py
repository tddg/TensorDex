#!/usr/bin/env python3
"""Build a delta-compressed TensorDex hub from models in the live S3 store,
following the SOSP'26 AE process (ingest -> FlexSplit plan -> TensorX
compress -> verified pull), then optionally upload the hub to another
bucket.

Per model:
  1. stage: stream-reconstruct the model from s3://tensor-tingfeng/tensordb
     blobs into a real .safetensors file (correct dtype/shape header,
     per-tensor XXH3-128 verification, bounded memory).
  2. ingest: hub.ingest() — Rust computes TensorID (dedup) + TensorSketch
     fingerprints and writes blobs + SQL.
  3. compress: `tensordex compress --auto <model> --include-existing-bases
     --codec tensorx --level 1` — FlexSplit attach planning over sketches,
     TensorPred ratio prediction, TensorX delta; blobs replaced in place,
     tensor_deltas recorded.
  4. verify: `tensordex pull --verify` (byte-exact hash check), output
     discarded.
  5. staging file deleted before the next model.

Usage:
  python eval/scripts/build_compressed_hub.py --hub eval/raw/compressed_hub \
      [--families small medium large] [--skip-verify]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SRC_BUCKET = "tensor-tingfeng"
SRC_PREFIX = "tensordb"
MASTER_DB = "eval/raw/master.db"

FAMILIES = {
    "small": [  # llama-3.2-3B, ~1.7 GB each (all 3 live variants)
        "AhmedSSoliman/llama-3.2-3b-chat-doctor",
        "Gabbar01/llama-3.2-3b-GSOC-DATASET",
        "BanglaLLM/BanglaLLama-3.2-3b-unolp-culturax-base-v0.0.1",
    ],
    "medium": [  # Qwen2.5-7B class, ~15.2 GB each
        "KPEP/krx-qwen-2.5-7b-v1.4.4",
        "HPAI-BSC/Qwen2.5-7B-Instruct-Egida-DPO",
        "mlfoundations-dev/hp_ablations_qwen_adambeta2_0.95_dcftv1.2",
    ],
    "large": [  # gemma-2-9B class, ~18.5 GB each; HEAD-preflighted —
        # jinoo/gemma2-9b dropped (417/464 blobs gc'd from the live store)
        "princeton-nlp/gemma-2-9b-it-SimPO",
        "TongZheng1999/gemma-2-9b-it-star-truth_table-OP-final_1-2-3Rounds-iter-1",
        "AlexBefest/WoonaV1.2-9b",
    ],
}

# torch dtype string (as stored in master.db) -> safetensors header dtype
DTYPE_MAP = {
    "torch.bfloat16": ("BF16", 2), "torch.float16": ("F16", 2),
    "torch.float32": ("F32", 4), "torch.float64": ("F64", 8),
    "torch.int64": ("I64", 8), "torch.int32": ("I32", 4),
    "torch.int16": ("I16", 2), "torch.int8": ("I8", 1),
    "torch.uint8": ("U8", 1), "torch.bool": ("BOOL", 1),
}

MAX_INFLIGHT_BYTES = 3 * 1024 ** 3
FETCH_WORKERS = 12


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def model_manifest(model: str):
    """[(param, tid, shape:list, dtype_str, nbytes)] from master.db."""
    con = sqlite3.connect(f"file:{MASTER_DB}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT mm.param_name, mm.tensor_id, t.shape, t.dtype, t.size_bytes
           FROM model_mappings mm JOIN tensors t ON t.id = mm.tensor_id
           WHERE mm.model_name = ? ORDER BY mm.rowid""", (model,)).fetchall()
    con.close()
    if not rows:
        raise SystemExit(f"model {model} not in {MASTER_DB}")
    return [(p, tid, json.loads(sh), dt, int(nb))
            for p, tid, sh, dt, nb in rows]


def stage_model(model: str, out_path: str, manifest=None) -> dict:
    """Stream-reconstruct `model` (or a subset of its params) into a valid
    safetensors file."""
    import boto3
    from botocore.config import Config

    from eval.src.adapters.tensordex import parse_safetensors_bytes
    from tensordex import _ops

    if manifest is None:
        manifest = model_manifest(model)

    # Header with real dtype/shape; offsets in manifest order.
    header: dict = {}
    off = 0
    for p, tid, shape, dt, nb in manifest:
        st_dtype, _ = DTYPE_MAP[dt]
        header[p] = {"dtype": st_dtype, "shape": shape,
                     "data_offsets": [off, off + nb]}
        off += nb
    total = off
    hjson = json.dumps(header, separators=(",", ":")).encode()
    pad = (8 - len(hjson) % 8) % 8  # keep payload 8-aligned
    hjson += b" " * pad
    base_off = 8 + len(hjson)

    # One fetch per unique tid; write to every offset that references it.
    by_tid: dict[str, list[tuple[int, int]]] = {}
    for p, tid, shape, dt, nb in manifest:
        b0 = header[p]["data_offsets"][0]
        by_tid.setdefault(tid, []).append((b0, nb))

    s3 = boto3.client("s3", config=Config(
        region_name="us-east-1", max_pool_connections=FETCH_WORKERS + 4,
        retries={"mode": "standard", "max_attempts": 5}))

    fd = os.open(out_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.pwrite(fd, struct.pack("<Q", len(hjson)) + hjson, 0)
    os.ftruncate(fd, base_off + total)

    inflight = {"bytes": 0}
    cond = threading.Condition()
    stats = {"fetched": 0, "payload": 0}

    def fetch_one(tid: str, size_hint: int):
        with cond:
            while inflight["bytes"] > MAX_INFLIGHT_BYTES:
                cond.wait()
            inflight["bytes"] += size_hint
        try:
            key = f"{SRC_PREFIX}/blobs/{tid[:2]}/{tid}.safetensors"
            body = s3.get_object(Bucket=SRC_BUCKET, Key=key)["Body"]
            data = body.read()
            _, _key, (_dt, _sh, payload) = parse_safetensors_bytes(data)
            if _ops.content_hash(payload) != tid:
                raise ValueError(f"hash mismatch for {tid}")
            for b0, nb in by_tid[tid]:
                if nb != len(payload):
                    raise ValueError(
                        f"size mismatch {tid}: db={nb} blob={len(payload)}")
                os.pwrite(fd, payload, base_off + b0)
            stats["fetched"] += 1
            stats["payload"] += len(payload)
        finally:
            with cond:
                inflight["bytes"] -= size_hint
                cond.notify_all()

    t0 = time.time()
    with cf.ThreadPoolExecutor(FETCH_WORKERS) as pool:
        futs = [pool.submit(fetch_one, tid, refs[0][1])
                for tid, refs in by_tid.items()]
        for f in cf.as_completed(futs):
            f.result()  # re-raise worker errors
    os.close(fd)
    dt_s = time.time() - t0
    log(f"  staged {total / 1e9:.2f} GB logical "
        f"({stats['payload'] / 1e9:.2f} GB fetched, {stats['fetched']} blobs, "
        f"{dt_s:.0f}s, {stats['payload'] / 1e9 / dt_s:.2f} GB/s)")
    return {"logical_bytes": total, "unique_blobs": len(by_tid),
            "payload_bytes": stats["payload"], "stage_seconds": dt_s}


def compress_model(hub, model: str, level: int) -> dict:
    """In-process auto_compress (subprocess CLI hits WAL visibility races
    against this process's open metadata connection)."""
    res = hub.auto_compress(model, cr_threshold=0.70, codec="tensorx",
                            level=level, include_existing_bases=True)
    results = res.get("results", [])
    ok = [r for r in results if r.get("status") == "ok"]
    orig = sum(r.get("original_bytes", 0) for r in ok)
    new = sum(r.get("compressed_bytes", 0) for r in ok)
    fails = [r for r in results if r.get("status") not in ("ok", "skipped")]
    log(f"  compressed {len(ok)}/{len(results)} pairs: "
        f"{orig / 1e6:.1f} MB -> {new / 1e6:.1f} MB"
        + (f", {len(fails)} FAILURES" if fails else ""))
    return {"pairs": len(results), "pairs_ok": len(ok),
            "orig_bytes": orig, "delta_bytes": new,
            "failures": len(fails)}


def verify_model_streaming(hub, model: str) -> dict:
    """Byte-exact verification with O(one pair) memory.

    hub.pull(verify=True) keeps every decoded base in an unbounded
    per-model cache — that OOM-killed a 15 GB model on this 16 GB box.
    Instead re-derive each unique tensor independently: raw tensors are
    re-hashed, deltas are decoded against their base (loaded fresh, then
    freed) and re-hashed. Same integrity guarantee as pull --verify.
    """
    import gc

    tids = sorted(set(hub.get_model_tensors(model).values()))
    for i, tid in enumerate(tids):
        t = hub.get_tensor(tensor_id=tid, verify=True)  # raises on mismatch
        del t
        if (i + 1) % 100 == 0:
            gc.collect()
    gc.collect()
    return {"verified_tensors": len(tids)}


def model_in_hub(hub, model: str, expected_params: int) -> bool:
    try:
        return len(hub.get_model_tensors(model)) == expected_params
    except Exception:  # noqa: BLE001
        return False


def hub_stats(hub: str) -> dict:
    con = sqlite3.connect(f"file:{hub}/metadata.db?mode=ro", uri=True)
    out = {
        "tensors": con.execute("SELECT COUNT(*) FROM tensors").fetchone()[0],
        "stored_bytes": con.execute(
            "SELECT COALESCE(SUM(size_bytes),0) FROM tensors").fetchone()[0],
        "deltas": con.execute(
            "SELECT COUNT(*) FROM tensor_deltas").fetchone()[0],
    }
    con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", default="eval/raw/compressed_hub")
    ap.add_argument("--staging", default="eval/raw/staging")
    ap.add_argument("--families", nargs="+", default=["small", "medium",
                                                      "large"])
    ap.add_argument("--level", type=int, default=1,
                    help="TensorX zstd level (paper trace used 1)")
    ap.add_argument("--chunk-bytes", type=int, default=4 * 1024 ** 3,
                    help="stage/ingest/compress granularity for big models")
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--report", default="eval/raw/compressed_hub_report.json")
    args = ap.parse_args()

    from tensordex import TensorDex

    os.makedirs(args.staging, exist_ok=True)
    hub = TensorDex(args.hub)
    report = {"models": {}, "logical_total": 0}

    for fam in args.families:
        for model in FAMILIES[fam]:
            log(f"=== [{fam}] {model}")
            rec = {"family": fam}
            staged = os.path.join(args.staging,
                                  model.replace("/", "__") + ".safetensors")

            manifest = model_manifest(model)
            if model_in_hub(hub, model, len(manifest)):
                rec["logical_bytes"] = sum(m[4] for m in manifest)
                rec["resumed"] = True
                report["logical_total"] += rec["logical_bytes"]
                log("  already ingested — skipping stage (resume)")
            else:
                # Chunked stage->ingest->compress: a large model must never
                # sit fully raw on disk (96 GB box). Each ~chunk of params
                # is staged, ingested, deleted, then attach-compressed so
                # raw bytes convert to deltas before the next chunk lands.
                chunks: list[list] = [[]]
                acc = 0
                for m in manifest:
                    if acc + m[4] > args.chunk_bytes and chunks[-1]:
                        chunks.append([])
                        acc = 0
                    chunks[-1].append(m)
                    acc += m[4]
                rec["logical_bytes"] = 0
                rec["ingest_seconds"] = 0.0
                rec["chunks"] = len(chunks)
                for ci, chunk in enumerate(chunks):
                    s = stage_model(model, staged, manifest=chunk)
                    rec["logical_bytes"] += s["logical_bytes"]
                    t0 = time.time()
                    res = hub.ingest([staged], model)
                    rec["ingest_seconds"] += time.time() - t0
                    os.remove(staged)
                    log(f"  chunk {ci + 1}/{len(chunks)}: ingested "
                        f"{len(res)} params")
                    if len(chunks) > 1:
                        compress_model(hub, model, args.level)
                got = len(hub.get_model_tensors(model))
                if got != len(manifest):
                    raise RuntimeError(
                        f"{model}: {got}/{len(manifest)} params after "
                        "chunked ingest")
                report["logical_total"] += rec["logical_bytes"]

            t0 = time.time()
            try:
                rec["compress"] = compress_model(hub, model, args.level)
            except Exception as e:  # noqa: BLE001
                rec["compress"] = {"error": repr(e)}
                log(f"  COMPRESS FAILED: {e!r}")
            rec["compress_seconds"] = time.time() - t0

            if not args.skip_verify:
                t0 = time.time()
                try:
                    v = verify_model_streaming(hub, model)
                    rec["verify_ok"] = True
                    log(f"  verified byte-exact "
                        f"({v['verified_tensors']} unique tensors, streaming)")
                except Exception as e:  # noqa: BLE001
                    rec["verify_ok"] = False
                    rec["verify_error"] = repr(e)
                    log(f"  VERIFY FAILED: {e!r}")
                rec["verify_seconds"] = time.time() - t0

            hs = hub_stats(args.hub)
            rec["hub_after"] = hs
            log(f"  hub: {hs['tensors']} tensors, "
                f"{hs['stored_bytes'] / 1e9:.2f} GB stored, "
                f"{hs['deltas']} deltas; cumulative logical "
                f"{report['logical_total'] / 1e9:.2f} GB "
                f"(reduction {1 - hs['stored_bytes'] / report['logical_total']:.1%})")
            report["models"][model] = rec
            with open(args.report, "w") as fh:
                json.dump(report, fh, indent=2)

    log("run `tensordex stats --hub ...` for the final breakdown; "
        f"report at {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
