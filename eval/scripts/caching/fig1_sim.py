#!/usr/bin/env python
"""Trace-driven cache simulation for the reconstruction-aware caching Figure 1.

Replays frozen, Zipf-weighted model-pull traces over the *real* compressed-hub
catalog (compressed_full/master.db + campaign per-pair ledger) and measures,
per policy x capacity cell:

  - direct hit ratio (requests / bytes)
  - origin bytes per pull, decomposed: delta-only misses (base resident),
    base+delta misses (base absent), raw misses
  - decode seconds per pull (measured decompress+hash proxy from the ledger)
  - physical base GETs vs logical base requirements (intra-pull dedup)

Policies: lru (hit-oriented, targets only), gdsf (static cost-aware prior
art), basebias (50% capacity pinned to top-value bases + LRU), recon
(reconstruction-aware: unified value-density over bases AND targets with
state-dependent accounting).

Inputs are read-only; outputs go to --out (JSON). Nothing touches S3.
"""
import argparse
import heapq
import json
import multiprocessing as mp
import os
import sqlite3
from collections import OrderedDict, defaultdict

import numpy as np

GB = 1024**3


# ---------------------------------------------------------------- profile ---

def build_profile(meta_dir):
    """Return catalog profile dicts built from master.db + exec ledgers."""
    db = sqlite3.connect(os.path.join(meta_dir, "master.db"))

    ledger = {}
    logdir = os.path.join(meta_dir, "logs")
    for fn in sorted(os.listdir(logdir)):
        if not fn.startswith("exec_done_"):
            continue
        with open(os.path.join(logdir, fn)) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") == "ok":
                    ledger[r["tid"]] = (r["orig"], r["comp"], r["verify_s"])

    deltas = dict(db.execute("SELECT tensor_id, base_tensor_id FROM tensor_deltas"))
    sizes = {}
    missing = set()
    for tid, lb, uri in db.execute("SELECT id, logical_bytes, storage_uri FROM tensors"):
        sizes[tid] = lb
        if uri and uri.startswith("missing://"):
            missing.add(tid)

    manifests = {}
    for model, tid in db.execute("SELECT model_name, tensor_id FROM model_mappings"):
        manifests.setdefault(model, []).append(tid)
    servable = {
        m: sorted(set(t))
        for m, t in manifests.items()
        if not any(x in missing for x in t)
    }

    med_ratio = float(np.median([c / o for o, c, _ in ledger.values() if o]))
    med_tput = float(np.median([o / v for o, _, v in ledger.values() if v > 0]))

    # integer-id world limited to tensors reachable from servable models
    tids, id_of = [], {}

    def intern(t):
        if t not in id_of:
            id_of[t] = len(tids)
            tids.append(t)
        return id_of[t]

    man_ids = {}
    for m, ts in servable.items():
        man_ids[m] = np.array([intern(t) for t in ts], dtype=np.int64)
        for t in ts:
            b = deltas.get(t)
            if b is not None and b not in missing:
                intern(b)

    n = len(tids)
    L = np.zeros(n, dtype=np.float64)        # logical (resident) size
    d = np.zeros(n, dtype=np.float64)        # origin delta size (0 for raw)
    c = np.zeros(n, dtype=np.float64)        # decode seconds (0 for raw)
    base = np.full(n, -1, dtype=np.int64)    # base id (-1 for raw)
    n_fallback = 0
    for i, t in enumerate(tids):
        L[i] = sizes[t]
        b = deltas.get(t)
        if b is not None and b not in missing and t in ledger:
            o, comp, vs = ledger[t]
            d[i], c[i], base[i] = comp, vs, id_of[b]
        elif b is not None and b not in missing:
            d[i], c[i], base[i] = med_ratio * L[i], L[i] / med_tput, id_of[b]
            n_fallback += 1

    return dict(tids=tids, L=L, d=d, c=c, base=base, manifests=man_ids,
                n_fallback=n_fallback, ledger=ledger)


def make_trace(models, seed, alpha, n_pulls):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(models))
    p = 1.0 / np.arange(1, len(models) + 1) ** alpha
    p /= p.sum()
    ranks = rng.choice(len(models), size=n_pulls, p=p)
    return [models[order[r]] for r in ranks]


# --------------------------------------------------------------- policies ---

class CacheBase:
    """Common bookkeeping: residency map + capacity enforcement."""

    def __init__(self, cap, prof):
        self.cap = cap
        self.prof = prof
        self.res = {}                       # id -> resident bytes
        self.used = 0.0

    def admit(self, i, size):
        if size > self.cap:
            return
        while self.used + size > self.cap:
            if not self.evict_one():
                return
        self.res[i] = size
        self.used += size

    def drop(self, i):
        self.used -= self.res.pop(i)


class LRU(CacheBase):
    """Hit-oriented: admit served tensors only; bases fetched transiently."""

    def __init__(self, cap, prof):
        super().__init__(cap, prof)
        self.order = OrderedDict()          # id -> None, LRU order

    def touch(self, i):
        if i in self.order:
            self.order.move_to_end(i)

    def on_serve(self, i, size):
        self.admit(i, size)
        if i in self.res:
            self.order[i] = None
            self.order.move_to_end(i)

    def on_base_fetch(self, b, size):
        pass

    def evict_one(self):
        while self.order:
            v, _ = self.order.popitem(last=False)
            if v in self.res:
                self.drop(v)
                return True
        return False


class GDSF(CacheBase):
    """GreedyDual-Size-Frequency; cost = origin bytes at last miss (static).

    Lazy min-heap: stale entries are skipped at pop time.
    """

    def __init__(self, cap, prof):
        super().__init__(cap, prof)
        self.clockv = 0.0
        self.freq = defaultdict(int)
        self.pri = {}
        self.last_cost = {}
        self.heap = []

    def note_miss_cost(self, i, cost):
        self.last_cost[i] = cost

    def _repri(self, i):
        p = self.clockv + self.freq[i] * (
            self.last_cost.get(i, self.res[i]) / max(self.res[i], 1.0))
        self.pri[i] = p
        heapq.heappush(self.heap, (p, i))

    def touch(self, i):
        self.freq[i] += 1
        if i in self.res:
            self._repri(i)

    def on_serve(self, i, size):
        self.admit(i, size)
        self.freq[i] += 1
        if i in self.res:
            self._repri(i)

    def on_base_fetch(self, b, size):
        pass

    def evict_one(self):
        while self.heap:
            p, v = heapq.heappop(self.heap)
            if v not in self.res or self.pri.get(v) != p:
                continue                     # stale entry
            self.clockv = max(self.clockv, p)
            del self.pri[v]
            self.drop(v)
            return True
        return False


class BaseBias(LRU):
    """Pin top-value bases in half the capacity; LRU targets in the rest."""

    def __init__(self, cap, prof, base_value):
        # base_value: id -> expected base-bytes avoided per pull (precomputed
        # from the trace-generator's popularity, i.e. an informed static pin)
        super().__init__(cap, prof)
        self.pinned = set()
        pin_budget = cap / 2
        for b in sorted(base_value, key=lambda x: -base_value[x] / prof["L"][x]):
            s = prof["L"][b]
            if self.used + s <= pin_budget:
                self.pinned.add(b)
                self.res[b] = s
                self.used += s

    def evict_one(self):
        while self.order:
            v, _ = self.order.popitem(last=False)
            if v in self.res and v not in self.pinned:
                self.drop(v)
                return True
        return False


class Recon(CacheBase):
    """Reconstruction-aware: unified EWMA value-density over bases+targets.

    value(target x)/byte  = freq_x * d_x / L_x          (origin bytes avoided)
    value(base b)/byte    = freq_family(b) + freq_b     (s_b avoided / s_b)
    freq_* is a decayed access count; eviction pops the minimum-density
    resident from a lazy heap (densities are recomputed at pop time and
    stale entries re-pushed). Bases are admitted on fetch, targets on serve.
    """
    DECAY = 0.999

    def __init__(self, cap, prof):
        super().__init__(cap, prof)
        self.freq = defaultdict(float)      # per-tensor decayed count
        self.bfreq = defaultdict(float)     # per-base-family decayed count
        self.heap = []

    def bump(self, i):
        self.freq[i] = self.freq[i] * self.DECAY + 1.0
        b = self.prof["base"][i]
        if b >= 0:
            self.bfreq[b] = self.bfreq[b] * self.DECAY + 1.0

    def density(self, i):
        p = self.prof
        if p["base"][i] == -1:               # raw tensor, maybe a base
            return self.bfreq.get(i, 0.0) + self.freq[i]
        return self.freq[i] * max(p["d"][i], 1.0) / max(p["L"][i], 1.0)

    def touch(self, i):
        self.bump(i)

    def on_serve(self, i, size):
        self.bump(i)
        self.admit(i, size)
        if i in self.res:
            heapq.heappush(self.heap, (self.density(i), i))

    def on_base_fetch(self, b, size):
        self.admit(b, size)
        if b in self.res:
            heapq.heappush(self.heap, (self.density(b), b))

    def evict_one(self):
        while self.heap:
            p, v = heapq.heappop(self.heap)
            if v not in self.res:
                continue
            cur = self.density(v)
            if cur > p * 1.05 + 1e-12:       # grew since push: re-file
                heapq.heappush(self.heap, (cur, v))
                continue
            self.drop(v)
            return True
        return False


# -------------------------------------------------------------- simulator ---

def run_cell(argsp):
    prof, trace, policy, cap_gb, warmup = argsp
    cap = cap_gb * GB
    L, d, c, base = prof["L"], prof["d"], prof["c"], prof["base"]

    if policy == "lru":
        cache = LRU(cap, prof)
    elif policy == "gdsf":
        cache = GDSF(cap, prof)
    elif policy == "basebias":
        # informed static pin: expected base-bytes avoided per pull under the
        # trace's own popularity (family pull-rate x s_b)
        counts = defaultdict(float)
        for man in trace:
            fams = {int(b) for b in np.unique(base[man]) if b >= 0}
            for b in fams:
                counts[b] += 1.0
        bv = {b: n / len(trace) * L[b] for b, n in counts.items()}
        cache = BaseBias(cap, prof, bv)
    elif policy == "recon":
        cache = Recon(cap, prof)
    else:
        raise ValueError(policy)

    m = dict(pulls=0, req=0, hit=0, req_bytes=0.0, hit_bytes=0.0,
             delta_only=0.0, base_bytes=0.0, delta_with_base=0.0,
             raw_bytes=0.0, decode_s=0.0, base_logical=0, base_physical=0)

    for k, man in enumerate(trace):
        live = k >= warmup
        fetched = set()
        for i in man:
            i = int(i)
            if live:
                m["req"] += 1
                m["req_bytes"] += L[i]
            if i in cache.res:
                cache.touch(i)
                if live:
                    m["hit"] += 1
                    m["hit_bytes"] += L[i]
                continue
            b = int(base[i])
            if b == -1:                                     # raw miss
                if live:
                    m["raw_bytes"] += L[i]
                if isinstance(cache, GDSF):
                    cache.note_miss_cost(i, L[i])
                cache.on_serve(i, L[i])
                continue
            # delta miss
            base_resident = b in cache.res or b in fetched
            if live:
                m["decode_s"] += c[i]
                m["base_logical"] += 1
                if base_resident:
                    m["delta_only"] += d[i]
                else:
                    m["delta_with_base"] += d[i]
            if b in cache.res:
                cache.touch(b)
            elif b not in fetched:
                fetched.add(b)
                if live:
                    m["base_bytes"] += L[b]
                    m["base_physical"] += 1
                cache.on_base_fetch(b, L[b])
            if isinstance(cache, GDSF):
                cache.note_miss_cost(i, d[i] + (0 if base_resident else L[b]))
            cache.on_serve(i, L[i])
        if live:
            m["pulls"] += 1

    n = max(m["pulls"], 1)
    origin = m["delta_only"] + m["delta_with_base"] + m["base_bytes"] + m["raw_bytes"]
    return dict(policy=policy, cap_gb=cap_gb,
                hit_ratio=m["hit"] / max(m["req"], 1),
                byte_hit_ratio=m["hit_bytes"] / max(m["req_bytes"], 1),
                origin_gb_per_pull=origin / n / GB,
                delta_only_gb=m["delta_only"] / n / GB,
                delta_with_base_gb=m["delta_with_base"] / n / GB,
                base_gb=m["base_bytes"] / n / GB,
                raw_gb=m["raw_bytes"] / n / GB,
                decode_s_per_pull=m["decode_s"] / n,
                base_logical=m["base_logical"] / n,
                base_physical=m["base_physical"] / n,
                logical_gb_per_pull=m["req_bytes"] / n / GB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True,
                    help="dir with master.db + logs/ (restored campaign meta)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pulls", type=int, default=12000)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caps", default="25,50,100,200,400")
    ap.add_argument("--policies", default="lru,gdsf,basebias,recon")
    args = ap.parse_args()

    prof = build_profile(args.meta)
    models = sorted(prof["manifests"])
    trace_models = make_trace(models, args.seed, args.alpha, args.pulls)
    trace = [prof["manifests"][m] for m in trace_models]
    print(f"universe: {len(models)} servable models, {len(prof['tids'])} tensors, "
          f"{prof['n_fallback']} ledger-fallback deltas", flush=True)

    slim = {k: prof[k] for k in ("L", "d", "c", "base")}
    cells = [(slim, trace, p, cap, args.warmup)
             for p in args.policies.split(",")
             for cap in [float(x) for x in args.caps.split(",")]]
    with mp.Pool(min(len(cells), mp.cpu_count() - 2)) as pool:
        results = pool.map(run_cell, cells)
    for r in results:
        print(f"{r['policy']:9s} cap={r['cap_gb']:6.0f}GB  hit={r['hit_ratio']:.3f} "
              f"origin={r['origin_gb_per_pull']:.2f}GB/pull "
              f"decode={r['decode_s_per_pull']:.1f}s/pull", flush=True)

    # catalog anatomy for panel (a)
    led = prof["ledger"]
    o = np.array([v[0] for v in led.values()], dtype=np.float64)
    comp = np.array([v[1] for v in led.values()], dtype=np.float64)
    r = comp / np.maximum(o, 1)
    fan = defaultdict(int)
    for i in range(len(prof["L"])):
        if prof["base"][i] >= 0:
            fan[int(prof["base"][i])] += 1
    anatomy = dict(
        ratio_q=np.quantile(r, np.linspace(0, 1, 201)).tolist(),
        ratio_q_bytes=weighted_quantiles(r, o, np.linspace(0, 1, 201)).tolist(),
        ratio_mean_w=float((r * o).sum() / o.sum()),
        fanin=sorted(fan.values(), reverse=True),
    )
    with open(args.out, "w") as f:
        json.dump(dict(config=vars(args), results=results, anatomy=anatomy), f)
    print("wrote", args.out)


def weighted_quantiles(x, w, qs):
    i = np.argsort(x)
    cw = np.cumsum(w[i])
    return np.interp(np.asarray(qs) * cw[-1], cw, x[i])


if __name__ == "__main__":
    main()
