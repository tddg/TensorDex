# TensorDex Cloud-Download Evaluation

Diagnostic evaluation for the FAST'27 decision study: where does end-to-end
model-download time go when TensorDex is backed by S3, compared against a
ZipNN-compressed artifact and raw `safetensors` downloads?

The full plan (research questions, metrics, stages, decision criteria) lives
in the project notes; this directory implements it.

## Layout

```
eval/
├── config/
│   ├── object_layout.yaml   # S3 prefixes for the three systems (Stage 0 output)
│   └── models.yaml          # frozen stratified model set (Stage 0 output)
├── scripts/
│   ├── inventory_s3.py      # list bucket, size by prefix -> eval/raw/
│   ├── inspect_metadata.py  # dump AE metadata.db / results.db schemas
│   └── ...                  # select_models / run_single / coordinator (later)
├── src/
│   ├── recorder.py          # append-only JSONL event recorder
│   ├── s3_tracing.py        # per-request GET/HEAD + StreamingBody timing
│   ├── metadata_tracing.py  # SQLite query-class instrumentation
│   ├── resource_monitor.py  # psutil sampler
│   └── adapters/            # tensordex / zipnn / raw DownloadAdapter impls
├── raw/                     # inventories, schema dumps, raw event logs
├── results/                 # single_client/ and multi_client/ outputs
└── figures/
```

## Environment status (2026-08-01)

- Instance: t3.xlarge, us-east-1d (NOT the c6a.48xlarge target machine —
  fine for Stage 0 correctness, not for performance runs).
- AWS credentials: **none present** (no instance profile, no ~/.aws).
  `s3://tensor-tingfeng/` denies anonymous access, so the S3 inventory and
  all download benchmarks are blocked until credentials are provisioned
  (instance profile or `AWS_PROFILE=tensordex-eval`). Never place keys in
  files or shell history.
- AE cache: downloaded from the public HF dataset
  `tensordex/tensordex-ae-cache` via `make ae-cache` (not from S3).

## Fairness rules (abridged from the plan)

Same instance type, bucket, region, model set, and output format for all
systems; fresh-process is the primary condition; verification excluded from
the primary timed interval; raw event logs always preserved; randomized
experiment order; versions and commits pinned in every run record.
