#!/usr/bin/env python3
"""Single-client benchmark driver (plan Stage 1, and Stage 0 Run 1/2).

For each (model, system) pair: fresh recorder, optional resource monitor,
resolve + download with phase events, and one summary line appended to
eval/results/single_client/model_runs.jsonl. Order of (model, system)
pairs is shuffled per repetition (fairness rule 9); pass --seed to fix it.

Usage:
    python eval/scripts/run_single.py --models model_a model_b \
        --systems tensordex zipnn raw --reps 3 --out eval/results/single_client
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import boto3  # noqa: E402
import yaml  # noqa: E402

from eval.src.recorder import EventRecorder  # noqa: E402
from eval.src.resource_monitor import ResourceMonitor  # noqa: E402
from eval.src.s3_tracing import TracedS3Client, make_config  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_adapter(system: str, layout: dict, models_cfg: dict,
                  s3c, args):
    if system in ("tensordex", "tensordex_c"):
        from eval.src.adapters.tensordex import TensorDexAdapter
        lay = layout["layouts"][system]
        adapter = TensorDexAdapter(
            s3c, layout["bucket"], lay["blob_prefix"],
            metadata_db_dir=lay["metadata_dir"],
            max_workers=args.workers)
        adapter.system = system
        return adapter
    if system == "tensordex_srv":
        from eval.src.adapters.tensordex_srv import TensorDexServerAdapter
        return TensorDexServerAdapter(
            metadata_url=args.metadata_url, cache_url=args.cache_url,
            max_workers=args.workers)
    if system == "zipnn":
        from eval.src.adapters.zipnn import ZipNNAdapter
        idx = {m: v["zipnn_objects"] for m, v in models_cfg.items()
               if "zipnn_objects" in v}
        return ZipNNAdapter(s3c, layout["bucket"],
                            layout["layouts"]["zipnn"]["model_prefix"], idx)
    if system == "raw":
        from eval.src.adapters.raw import RawAdapter
        idx = {m: v["raw_objects"] for m, v in models_cfg.items()
               if "raw_objects" in v}
        return RawAdapter(s3c, layout["bucket"],
                          layout["layouts"]["raw"]["model_prefix"], idx)
    raise ValueError(system)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--systems", nargs="+",
                    default=["tensordex", "zipnn", "raw"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--layout", default="eval/config/object_layout.yaml")
    ap.add_argument("--models-config", default="eval/config/models.yaml")
    ap.add_argument("--metadata-dir",
                    default="ae/cache/data/tensordb_s3")
    ap.add_argument("--out", default="eval/results/single_client")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--pool", type=int, default=256)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--keep-output", action="store_true")
    ap.add_argument("--metadata-url", default="http://127.0.0.1:8701")
    ap.add_argument("--cache-url", default="http://127.0.0.1:8702")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (model, system, rep) combos already ok in "
                         "model_runs.jsonl (resume after a crash)")
    args = ap.parse_args()

    with open(args.layout) as fh:
        layout = yaml.safe_load(fh)
    models_cfg = {}
    if os.path.exists(args.models_config):
        with open(args.models_config) as fh:
            models_cfg = yaml.safe_load(fh) or {}

    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    runs_path = os.path.join(args.out, "model_runs.jsonl")

    done: set[tuple] = set()
    if args.skip_existing and os.path.exists(runs_path):
        with open(runs_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("ok"):
                    done.add((r["model_id"], r["system"], r["rep"]))
        print(f"resume: {len(done)} completed runs will be skipped",
              flush=True)

    env_record = {
        "instance_type": os.environ.get("EVAL_INSTANCE_TYPE", "unknown"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": git_commit(),
        "workers": args.workers,
        "pool": args.pool,
        "argv": sys.argv,
    }

    for rep in range(args.reps):
        pairs = [(m, s) for m in args.models for s in args.systems]
        rng.shuffle(pairs)
        for model_id, system in pairs:
            if (model_id, system, rep) in done:
                print(f"[{rep}] {system:10s} {model_id:40s} skipped (done)",
                      flush=True)
                continue
            run_id = f"{system}-{model_id.replace('/', '_')}-r{rep}-" \
                     f"{int(time.time())}"
            events_path = os.path.join(args.out, f"events_{run_id}.jsonl")
            outdir = os.path.join(args.out, "output", run_id)
            raw_client = boto3.client("s3", config=make_config(
                max_pool_connections=args.pool, region=layout["region"]))
            rec = EventRecorder(events_path, run_id=run_id,
                                system=system, model_id=model_id)
            s3c = TracedS3Client(raw_client, rec)
            adapter = build_adapter(system, layout, models_cfg, s3c, args)
            mon = None if args.no_monitor else ResourceMonitor(rec).start()

            t0 = time.perf_counter_ns()
            rec.emit("metadata_query_start", phase="resolve")
            try:
                plan = adapter.resolve(model_id)
                rec.emit("metadata_query_end", phase="resolve",
                         n_objects=len(plan.objects))
                result = adapter.download(plan, outdir, rec)
            except Exception as e:  # noqa: BLE001
                from eval.src.adapters.base import DownloadResult
                result = DownloadResult(model_id=model_id, system=system,
                                        ok=False, error=repr(e))
            t1 = time.perf_counter_ns()
            if mon:
                mon.stop()
            rec.close()

            row = {
                "run_id": run_id, "rep": rep, "system": system,
                "model_id": model_id,
                "model_e2e_ms": (t1 - t0) / 1e6,
                "ok": result.ok, "error": result.error,
                "logical_model_bytes": result.logical_model_bytes,
                "s3_payload_bytes": result.s3_payload_bytes,
                "s3_get_count": result.s3_get_count,
                "events": events_path,
                "env": env_record,
            }
            with open(runs_path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            if args.keep_output:
                pass
            elif os.path.isdir(outdir):
                # Disk cannot hold accumulating 15-18 GB outputs; the
                # per-tensor hash verify already proved byte-exactness.
                import shutil
                shutil.rmtree(outdir)
            status = "ok" if result.ok else f"FAILED: {result.error}"
            print(f"[{rep}] {system:10s} {model_id:40s} "
                  f"{(t1 - t0) / 1e9:8.2f}s  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
