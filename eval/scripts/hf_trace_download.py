#!/usr/bin/env python3
"""Traced Hugging Face downloads: record WHERE every byte comes from.

For each repo: list weight files, then GET each via /resolve following
redirects manually, recording every hop's host + provenance headers
(via / x-cache / x-amz-cf-pop -> CloudFront; x-hf-cdn-pop -> HF's own
CDN; server: AmazonS3 -> direct S3), then stream the body to /dev/null
with TTFB + throughput per request. hf-CLI-like parallelism (8 files
in flight per model). Output: JSONL, one record per request.
"""
import argparse
import concurrent.futures as cf
import json
import socket
import ssl
import time
import urllib.parse
import urllib.request

CHUNK = 8 * 1024 * 1024
HDRS_OF_INTEREST = [
    "server", "via", "x-cache", "x-amz-cf-pop", "x-amz-cf-id", "age",
    "x-hf-cdn-pop", "x-hub-cache", "x-request-id", "content-length",
    "x-linked-size", "etag",
]


def classify(host, headers):
    h = host.lower()
    if "cloudfront" in (headers.get("via") or "") or "cloudfront.net" in h:
        xc = (headers.get("x-cache") or "").lower()
        return "cloudfront-hit" if "hit" in xc else "cloudfront-miss"
    if headers.get("x-hf-cdn-pop") or h.endswith(".cdn.hf.co"):
        return "hf-own-cdn"
    if ".s3." in h or h.endswith("amazonaws.com"):
        return "s3-direct"
    if h.endswith("huggingface.co"):
        return "hub"
    return "other:" + h


def list_weights(repo):
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=30) as r:
        files = json.load(r)
    return [(f["path"], f["size"]) for f in files
            if f["path"].endswith(".safetensors") and f.get("size")]


def fetch_traced(repo, path, size, rep, out_lock, out_f, rng=None):
    url = ("https://huggingface.co/" + repo + "/resolve/main/"
           + urllib.parse.quote(path))
    hops = []
    t_start = time.monotonic()
    for _ in range(6):
        req = urllib.request.Request(url, method="GET")
        if rng is not None:
            req.add_header("Range", f"bytes={rng[0]}-{rng[1]}")
        opener = urllib.request.build_opener(NoRedirect())
        t0 = time.monotonic()
        try:
            resp = opener.open(req, timeout=300)
        except urllib.error.HTTPError as e:
            resp = e
        host = urllib.parse.urlparse(url).netloc
        hdrs = {k: resp.headers.get(k) for k in HDRS_OF_INTEREST
                if resp.headers.get(k)}
        hop = {"host": host, "status": resp.status,
               "cls": classify(host, {k.lower(): v for k, v in
                                      resp.headers.items()}),
               "hdr": hdrs, "hop_ms": round((time.monotonic() - t0) * 1e3, 1)}
        hops.append(hop)
        if resp.status in (301, 302, 303, 307, 308):
            url = urllib.parse.urljoin(url, resp.headers["Location"])
            resp.close()
            continue
        # final response: stream body
        ttfb = None
        got = 0
        t1 = time.monotonic()
        while True:
            b = resp.read(CHUNK)
            if ttfb is None:
                ttfb = time.monotonic() - t1
            if not b:
                break
            got += len(b)
        t2 = time.monotonic()
        resp.close()
        rec = {"repo": repo, "path": path, "rep": rep, "size": size,
               "range": list(rng) if rng else None,
               "bytes": got, "hops": hops,
               "final_cls": hop["cls"], "final_host": host,
               "ttfb_ms": round(ttfb * 1e3, 1) if ttfb is not None else None,
               "stream_s": round(t2 - t1, 3),
               "total_s": round(t2 - t_start, 3),
               "MiBps": round(got / 2**20 / max(t2 - t1, 1e-6), 1)}
        with out_lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
        return rec
    raise RuntimeError(f"too many redirects: {repo}/{path}")


class NoRedirect(urllib.request.HTTPErrorProcessor):
    def http_response(self, req, resp):
        return resp
    https_response = http_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="+", required=True)
    ap.add_argument("--rep2", nargs="*", default=[],
                    help="repos to fetch a second time (edge-warmth probe)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import threading
    lock = threading.Lock()
    with open(args.out, "a") as out_f:
        PART = 128 * 2**20   # split big files into 128 MiB range-parts so
        for repo in args.repos:  # concurrency matches hf-CLI-style transfer
            reps = [1, 2] if repo in args.rep2 else [1]
            try:
                files = list_weights(repo)
            except Exception as exc:
                with lock:
                    out_f.write(json.dumps({
                        "repo": repo, "model_summary": True,
                        "error": repr(exc)}) + "\n")
                    out_f.flush()
                print(f"{repo}: SKIPPED ({exc})", flush=True)
                continue
            parts = []
            for p, s in files:
                if s <= PART:
                    parts.append((p, s, None))
                else:
                    for off in range(0, s, PART):
                        parts.append((p, s, (off, min(off + PART, s) - 1)))
            for rep in reps:
                t0 = time.monotonic()
                with cf.ThreadPoolExecutor(args.workers) as ex:
                    futs = [ex.submit(fetch_traced, repo, p, s, rep,
                                      lock, out_f, rng)
                            for p, s, rng in parts]
                    recs = [f.result() for f in futs]
                wall = time.monotonic() - t0
                got = sum(r["bytes"] for r in recs)
                with lock:
                    out_f.write(json.dumps({
                        "repo": repo, "rep": rep, "model_summary": True,
                        "files": len(files), "parts": len(parts),
                        "bytes": got,
                        "wall_s": round(wall, 2),
                        "MiBps": round(got / 2**20 / wall, 1),
                        "classes": sorted({r["final_cls"] for r in recs}),
                    }) + "\n")
                    out_f.flush()
                print(f"{repo} rep{rep}: {got/2**30:.2f} GiB in {wall:.1f}s "
                      f"= {got/2**20/wall:.0f} MiB/s "
                      f"[{','.join(sorted({r['final_cls'] for r in recs}))}]",
                      flush=True)


if __name__ == "__main__":
    main()
