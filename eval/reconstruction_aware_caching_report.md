# Reconstruction-Aware Caching for TensorDex

## Problem framing, analytical model, motivating example, and a signature Figure 1

## Executive summary

TensorDex’s hot cache differs from an ordinary object cache because a cache miss does not have a fixed cost. A reconstructed target tensor can be served from a compressed delta only when its raw base tensor is already resident. Otherwise, the cache must fetch both the base and the delta and then decode the target.

For a target tensor of logical size \(L_x\), let \(r_x=d_x/L_x\) be its tensor-specific delta ratio. The expected average is approximately \(0.33\)–\(0.34\), but the empirical distribution spans a long range. Thus the familiar \(0.34L\) and \(1.34L\) costs are useful averages, not fixed constants:

- **Materialization hit:** \(0\) origin bytes and \(0\) decodes.
- **Target miss with resident base:** \(d_x=r_xL_x\) origin bytes and \(1\) decode.
- **Target miss without resident base:** \(d_x+s_{b(x)}\) origin bytes and \(1\) decode; this is approximately \((1+r_x)L_x\) only when base and target sizes are similar.
- **Raw tensor miss:** its raw size from origin and \(0\) decodes.

When several missing targets require the same absent base, the cache fetches that base **once** and shares the result across all of those reconstructions. This sharing applies within a model pull and should also be coalesced across overlapping in-flight pulls.

The cache therefore chooses between two qualitatively different uses of space:

1. **Cache reconstructed targets.** This creates ordinary hits, avoiding both origin traffic and decode work for those targets.
2. **Cache raw bases.** This may create no ordinary target hits, but it converts future misses across many descendants from expensive \(1.34L\) misses into cheap \(0.34L\) misses.

This causes two central effects:

- **Hit ratio can rank policies incorrectly.** A base-heavy policy may have a lower direct hit ratio while moving far fewer origin bytes.
- **The best policy depends on the current system bottleneck.** A base-heavy allocation is preferable when S3 bandwidth is scarce; a materialization-heavy allocation is preferable when decode CPU is scarce.

The proposed FAST’27 problem should therefore be framed as **reconstruction-aware caching with state-dependent, shared, and multi-resource miss costs**, rather than generic “DAG-aware caching.”

---

## 1. System setting

The cache fronts an S3 origin containing:

- raw, uncompressed tensors that may also serve as reconstruction bases; and
- compressed deltas from which target tensors can be reconstructed.

The hot cache stores materialized tensors. A resident raw tensor can simultaneously play two roles:

- a normal servable tensor, if a model directly requests it; and
- a reconstruction prerequisite for many target tensors.

A model pull requests a bundle of hundreds of tensors. The cache must decide which reconstructed targets and which bases deserve residency under a fixed byte capacity.

The key semantic difference from ordinary caching is that a target’s miss cost depends on the residency of another object.

---

## 2. Why the problem is not ordinary cost-aware caching

Classic size- or cost-aware policies assign each object an independent retrieval cost and estimate its expected benefit per cache byte. TensorDex violates that assumption in several interacting ways.

### 2.1 State-dependent miss cost

For a delta-compressed target \(x\) with base \(b(x)\), logical size \(L_x\), and delta size \(d_x\):

\[
\mathrm{MissBytes}(x,S)
=
d_x + s_{b(x)}\mathbf{1}[b(x)\notin S],
\]

where \(S\) is the current cache state.

The same target miss can therefore be cheap or expensive depending on whether its base is resident. Using \(0.34L\) and \(1.34L\) is only an average-case illustration.

### 2.2 Heterogeneous compression changes object value

Define the tensor-specific compression ratio:

\[
r_x=\frac{d_x}{L_x}.
\]

Although \(\mathbb{E}[r_x]\approx 0.33\)–\(0.34\), \(r_x\) has a long-range distribution. This adds another layer of coupling:

- a highly compressible target may be inexpensive to miss when its base is resident;
- a poorly compressible target may be worth materializing even if its request frequency is moderate;
- two descendants of the same base can have very different marginal values;
- the bandwidth–CPU crossover is family- and tensor-specific rather than a single global threshold;
- a policy based on average compression ratio can systematically mis-rank the tails, which may dominate bytes because tensors are also variable-sized.

The policy should therefore use measured \(d_x\), \(L_x\), and decode cost \(c_x\), not a single global ratio.

### 2.3 Shared prerequisite value

A base may protect misses for hundreds or thousands of descendants. It can be valuable even when it is rarely or never directly requested.

A conventional hit counter does not recognize this value because the base creates a **prerequisite hit**, not a direct materialization hit.

### 2.4 One base fetch serves multiple delta misses

A model pull contains many co-requested tensors. If several missing targets require the same absent base, the origin should return that base once, after which the cache can reconstruct all of those targets.

Consequently, base bytes are charged once per distinct required base:

\[
s_b \cdot
\mathbf{1}
\left[
\exists x \in q:
x\notin S,\ b(x)=b
\right],
\]

rather than once per target:

\[
\sum_{x \in q:b(x)=b}
s_b\mathbf{1}[x\notin S].
\]

The same principle should apply across concurrent requests. If multiple pulls miss on descendants of the same base while its fetch is in progress, the materializer should maintain a single-flight entry and attach all waiters to one origin request. Otherwise the experiment measures duplicate-fetch races rather than the intended cache policy.

This sharing creates two implications:

1. **Bundle-aware valuation:** a base's benefit depends on the probability that at least one uncached descendant is requested in a bundle, not on the sum of descendant frequencies.
2. **Concurrency-aware service cost:** request overlap affects S3 GET count, queueing, and latency even when total logical demand is unchanged.

### 2.5 Different objects save different resources

Caching a reconstructed target avoids:

- its delta transfer;
- a possible base transfer;
- one decode.

Caching its base avoids:

- future base transfers across descendants;

but it does **not** avoid target decoding.

The cache is therefore allocating capacity between bandwidth-saving objects and compute-saving objects.

---

## 3. Formal request-cost model

Let:

- \(B\) be raw tensors;
- \(D\) be delta-compressed targets;
- \(b(x)\in B\) be the base of target \(x\in D\);
- \(s_z\) be the resident size of object \(z\);
- \(d_x\) be the origin delta size of target \(x\);
- \(c_x\) be the decode CPU cost of target \(x\);
- \(S\) be the current cache contents;
- \(q\) be a model-pull request containing a bundle of tensors.

The missed delta targets are:

\[
M_D(q,S)=\{x\in q\cap D:x\notin S\}.
\]

The raw tensors that must be fetched are:

\[
M_B(q,S)=
\left\{
b\in B\setminus S:
b\in q
\ \lor\
\exists x\in M_D(q,S),\ b(x)=b
\right\}.
\]

The origin bytes incurred by request \(q\) are:

\[
\mathrm{Bytes}(q,S)
=
\sum_{x\in M_D(q,S)}d_x
+
\sum_{b\in M_B(q,S)}s_b.
\]

Because \(M_B(q,S)\) is a set, each distinct base contributes its bytes at most once no matter how many missed descendants use it. In implementation, an in-flight fetch table should extend this uniqueness across overlapping requests:

\[
\mathrm{OriginFetches}(b,\Delta t)\le 1
\]

for all misses that arrive while the first fetch of \(b\) remains in progress.

Compression heterogeneity is preserved exactly through the per-target \(d_x\) term. No average ratio is required by the model.

The reconstruction work is:

\[
\mathrm{CPU}(q,S)
=
\sum_{x\in M_D(q,S)}c_x.
\]

A weighted service-cost objective is:

\[
J(q,S,t)
=
\lambda_B(t)\mathrm{Bytes}(q,S)
+
\lambda_C(t)\mathrm{CPU}(q,S),
\]

where:

- \(\lambda_B(t)\) is the current shadow price of origin bandwidth;
- \(\lambda_C(t)\) is the current shadow price of decode capacity.

The weights may change over time as bottlenecks shift.

A systems-oriented alternative is to minimize bottleneck-normalized demand:

\[
\rho(S,t)=
\max\left(
\frac{\lambda(t)\mathbb{E}[\mathrm{Bytes}(q,S)]}{B_{\mathrm{S3}}(t)},
\frac{\lambda(t)\mathbb{E}[\mathrm{CPU}(q,S)]}{C_{\mathrm{decode}}(t)}
\right),
\]

where \(\lambda(t)\) is the request rate.

This objective directly connects cache allocation to sustainable throughput.

---

## 4. Minimal motivating example: intuition, not the paper figure

Consider a family with:

- one raw base \(b\);
- two equally popular reconstructed targets \(x\) and \(y\);
- each resident object occupying \(L\) bytes;
- each target delta occupying \(rL=0.34L\);
- one decode required for every target miss;
- cache capacity exactly \(L\).

The cache can retain either the shared base or one reconstructed target.

### 4.1 Policy A: cache the base

Cache \(b\).

Every request for \(x\) or \(y\) is a direct target miss, but the base is resident.

Per request:

\[
\mathrm{origin}=0.34L,
\]

\[
\mathrm{decodes}=1,
\]

\[
\mathrm{direct\ target\ hit\ ratio}=0\%.
\]

### 4.2 Policy B: cache one reconstructed target

Cache \(x\). Requests are evenly split between \(x\) and \(y\).

- A request for \(x\) is a hit.
- A request for \(y\) fetches \(b\) and \(d_y\), costing \(1.34L\), then decodes \(y\).

Average per request:

\[
\mathrm{origin}
=
0.5\times 0
+
0.5\times 1.34L
=
0.67L,
\]

\[
\mathrm{decodes}
=
0.5,
\]

\[
\mathrm{direct\ target\ hit\ ratio}=50\%.
\]

### 4.3 The inversion

| Allocation | Direct target hit ratio | Origin bytes/request | Decodes/request |
|---|---:|---:|---:|
| Cache shared base | 0% | \(0.34L\) | 1.0 |
| Cache one target | 50% | \(0.67L\) | 0.5 |

The higher-hit-ratio policy moves almost twice as many origin bytes. The lower-hit-ratio policy performs twice as many decodes.

Therefore:

- **S3-constrained system:** cache the base.
- **CPU-constrained system:** cache the reconstructed target.

This example captures the central inversion in the smallest possible setting, but it is intentionally too simple to serve as the paper's signature Figure 1. It assumes equal sizes, equal popularity, one common compression ratio, one request at a time, and only two legal placements. The actual figure should reveal the same inversion under a heterogeneous, bundled, concurrent workload drawn from the real TensorDex catalog.

---

## 5. Analytical bottleneck crossover

Let:

- \(p\) be the request probability of the most popular target;
- \(r=d/L\) be the delta ratio;
- \(R=B_{\mathrm{S3}}/L\) be the number of raw-tensor equivalents that S3 can deliver per second;
- \(D=C_{\mathrm{decode}}/c\) be the number of decodes the cache can perform per second.

With one-object cache capacity:

### Cache the shared base

\[
\mathrm{Bytes/request}=rL,
\]

\[
\mathrm{Decodes/request}=1.
\]

Sustainable throughput is:

\[
\mu_{\mathrm{base}}
=
\min\left(
\frac{R}{r},
D
\right).
\]

### Cache the most popular reconstructed target

\[
\mathrm{Bytes/request}
=
(1-p)(1+r)L,
\]

\[
\mathrm{Decodes/request}
=
1-p.
\]

Sustainable throughput is:

\[
\mu_{\mathrm{target}}
=
\min\left(
\frac{R}{(1-p)(1+r)},
\frac{D}{1-p}
\right).
\]

For \(p=0.5\) and \(r=0.34\):

\[
\mu_{\mathrm{base}}
=
\min\left(
\frac{R}{0.34},
D
\right),
\]

\[
\mu_{\mathrm{target}}
=
\min\left(
\frac{R}{0.67},
2D
\right).
\]

Define the resource ratio:

\[
\phi=\frac{R}{D}.
\]

The throughput crossover occurs at:

\[
\phi^\star=(1-p)(1+r)=0.67.
\]

Thus:

- when \(\phi<0.67\), origin bandwidth is relatively scarce and base caching wins;
- when \(\phi>0.67\), decode capacity is relatively scarce and target caching wins.

This crossover is not a property of popularity alone. It depends jointly on workload and current resource supply. With real tensors, replace the single \(r\) by measured \(r_x\), account for variable base and target sizes, charge a shared base once per request bundle, and include observed decode costs. The resulting workload has a *distribution of local crossovers* rather than one universal threshold.

---

## 6. Proposed solution direction: bottleneck-aware marginal value

A principled policy should estimate the increase in expected service cost caused by evicting resident object \(z\):

\[
V(z\mid S,t)
=
\mathbb{E}_q
\left[
J(q,S\setminus\{z\},t)-J(q,S,t)
\right].
\]

Its eviction density is:

\[
\mathrm{score}(z)
=
\frac{V(z\mid S,t)}{s_z}.
\]

The policy evicts objects with the smallest marginal value per resident byte.

The estimator must use per-tensor delta sizes and decode costs. It should also evaluate bases over distinct request bundles and coalesced in-flight fetches; simply multiplying a base's size by descendant miss count double-counts shared origin work.

### 6.1 Base value

The bandwidth value of base \(b\) is approximately:

\[
V_B(b\mid S,t)
=
\lambda_B(t)s_b
\sum_q
\hat{\lambda}_q
\mathbf{1}
\left[
b\in q
\ \lor\
\exists x\in q\setminus S:b(x)=b
\right].
\]

Important details:

- The indicator is evaluated once per request bundle.
- Resident descendants generally reduce the prerequisite value of the base.
- A directly requested base contributes ordinary hit value as well.

### 6.2 Reconstructed-target value

A resident target \(x\) avoids:

- its delta transfer \(d_x\);
- one decode \(c_x\);
- sometimes the entire base transfer, when it is the final nonresident descendant requiring that base in the request.

An approximate target value is:

\[
V_T(x\mid S,t)
=
\sum_{q:x\in q}\hat{\lambda}_q
\left[
\lambda_B(t)d_x
+
\lambda_C(t)c_x
+
\lambda_B(t)s_{b(x)}I_{\mathrm{last}}(x,q,S)
\right].
\]

\(I_{\mathrm{last}}\) is one when caching \(x\) eliminates the final need to fetch \(b(x)\) for request \(q\).

### 6.3 Dynamic resource prices

The policy should adapt \(\lambda_B(t)\) and \(\lambda_C(t)\) using recent system pressure.

Possible signals include:

- S3-fetch queue length;
- measured origin-byte service time;
- S3 throughput utilization;
- decode queue length;
- decode-worker utilization;
- decode waiting time;
- p95 contribution of origin and decode stages to request latency.

A simple implementation can update the weights every few seconds:

\[
\lambda_B(t+1)
=
\lambda_B(t)
\exp\left(
\eta(u_B(t)-u_B^\star)
\right),
\]

\[
\lambda_C(t+1)
=
\lambda_C(t)
\exp\left(
\eta(u_C(t)-u_C^\star)
\right),
\]

where \(u_B^\star\) and \(u_C^\star\) are target utilizations.

This behaves like an online dual update:

- S3 congestion increases the value of bases and other bandwidth-saving objects.
- Decode congestion increases the value of materialized targets.
- The cache composition shifts as the bottleneck changes.

---

# 7. Signature Figure 1: reveal the inversion in a realistic workload

## Proposed title

**Figure 1: More cache hits can cost more—and the best cache contents change with the bottleneck.**

The minimal two-target example should appear in the text or a sidebar, not as the main figure. The signature figure should be built from a nontrivial TensorDex workload that simultaneously includes:

- hundreds of tensors per model pull;
- multiple bases and many descendants per base;
- heterogeneous tensor sizes;
- a long-range distribution of delta ratios;
- skewed and time-varying model popularity;
- overlapping model families;
- several misses in one pull sharing a base;
- concurrent pulls sharing an in-flight base fetch;
- a bottleneck that changes between S3 bandwidth and decode CPU.

The figure should make three claims visible from measured data:

1. Direct target hit ratio is not monotonic with origin cost or model-pull latency.
2. Base residency and materialization residency trade S3 work against decode work differently.
3. A policy that is best in one resource regime can become worse after the bottleneck changes, even with the same cache size and largely the same object popularity.

---

## 7.1 Construct a representative workload rather than a toy family

Build the Figure 1 trace from a selected subgraph of the real TensorDex catalog. The subgraph should be large enough to exercise real interactions but small enough to explain visually and run repeatedly.

### Recommended subgraph

Select approximately:

- **20–50 model repositories**;
- **5,000–20,000 target tensors**;
- **50–200 active bases**;
- **several high-fan-out bases**, several medium-fan-out bases, and some nearly unattached raw tensors;
- target delta ratios spanning multiple quantiles of the measured distribution, for example the 10th, 25th, 50th, 75th, and 90th percentiles;
- a mixture of small and large tensors so bytes are not proportional to request count.

The subgraph should preserve complete model manifests. Do not sample isolated tensors, because doing so destroys co-access and shared-base behavior.

### Real-world scenario sequence

Compose a trace with three recognizable phases:

#### Phase A: family-release burst

A newly released base model produces many fine-tunes or variants that are pulled in a short interval. Requests span many siblings, so the same bases support many different target misses.

Expected behavior:

- direct target reuse is modest;
- prerequisite reuse is high;
- multiple deltas in one model pull share a base;
- overlapping pulls frequently coalesce on an in-flight base fetch;
- S3 bandwidth is the dominant pressure;
- base-heavy or hybrid residency should outperform materialization-heavy caching despite a lower direct hit ratio.

#### Phase B: steady deployment of a few hot variants

A small set of models becomes repeatedly deployed or autoscaled. Requests concentrate on the same reconstructed tensors.

Expected behavior:

- target reuse is high;
- repeatedly decoding the same targets becomes wasteful;
- materialization-heavy caching raises direct hit ratio and reduces decode demand;
- decode CPU can become the bottleneck even if origin traffic remains moderate.

#### Phase C: mixed hub churn

A background stream of long-tail pulls coexists with popular deployments and occasional family bursts.

Expected behavior:

- neither all-base nor all-target placement is robust;
- delta-ratio heterogeneity matters: low-ratio descendants are cheap to reconstruct when their bases are resident, while high-ratio or expensive-to-decode targets deserve materialization;
- a bottleneck-aware hybrid policy should allocate by measured marginal value.

These phases resemble plausible model-hub behavior without claiming that the exact trace is a production trace. The paper should label it **catalog-grounded, scenario-driven** unless real request logs are available.

---

## 7.2 Candidate policies and placements

Figure 1 should compare at least four cache behaviors under the same capacity:

1. **Hit-oriented:** LRU or TinyLFU-style admission over reconstructed targets.
2. **Base-biased:** workload-weighted retention of high-value bases, with remaining space for targets.
3. **Static cost-aware:** GDSF/LHD-style scoring using fixed or average miss cost.
4. **Reconstruction- and bottleneck-aware:** per-tensor sizes, shared-base accounting, bundle co-access, and dynamic bandwidth/CPU prices.

For causal clarity, also evaluate two offline static placements for each trace window:

- a placement optimized only for direct target hits;
- a placement optimized for origin bytes under the measured request bundles.

These are explanatory anchors, not deployable baselines.

---

## 7.3 Recommended four-panel layout

A four-panel layout can remain compact if each panel has one job.

### Panel (a): Workload anatomy and shared reconstruction

Show a small visual summary of the selected real subgraph rather than the entire DAG:

- several bases with visibly different fan-outs;
- model-pull bundles crossing multiple bases;
- descendants labeled with different delta ratios, such as \(0.08\), \(0.31\), \(0.55\), and \(0.92\);
- one request containing several deltas that share a base;
- two concurrent requests joining one in-flight base fetch.

Alongside the schematic, include two tiny empirical distributions from the selected workload:

- histogram or ECDF of \(r_x=d_x/L_x\);
- fan-out or shared-base multiplicity distribution.

The visual message should be:

> Miss costs vary across tensors, and one base fetch can serve many misses.

### Panel (b): Hit-ratio inversion on a real trace

For all policy-capacity points, plot:

- **x-axis:** direct reconstructed-target hit ratio;
- **y-axis:** origin bytes per completed model pull;
- **point size:** decode core-seconds per pull;
- **point shape:** policy;
- **optional outline:** workload phase.

Annotate at least one measured pair with the same cache capacity:

- Point H: higher direct hit ratio, but higher origin bytes because misses require absent bases plus deltas.
- Point B: lower direct hit ratio, but lower origin bytes because resident bases turn many target misses into cheap delta-only misses.

Connect the pair with an arrow labeled:

> **More hits, more S3 traffic.**

This panel is stronger than a two-bar toy example because it shows that the inversion persists across a heterogeneous workload and is not an artifact of equal sizes or two objects.

A useful headline statistic can appear in the panel or caption:

> Across all equal-capacity policy pairs, direct hit-ratio ordering disagrees with origin-byte ordering in \(X\%\) of comparisons.

The value \(X\) must be measured, not assumed.

### Panel (c): Resource-regime phase diagram

Construct a two-dimensional sweep:

- **x-axis:** effective S3 bandwidth;
- **y-axis:** available decode capacity, such as decode workers or calibrated decodes/s.

At each grid point, replay the same workload window and identify the best measured policy or cache composition by sustainable throughput or p99 latency under a fixed offered load.

Color each cell by either:

- winning policy; or
- optimal fraction of cache bytes assigned to bases.

Overlay contours of equal throughput and mark the measured operating points used in Panels (b) and (d).

Expected regions:

- low S3 / ample CPU: base-heavy;
- high S3 / scarce CPU: target-heavy;
- intermediate resources: hybrid;
- heterogeneous workload corners where per-tensor delta ratios create nontrivial boundaries rather than a straight analytical line.

The phase boundary is itself an important result: it demonstrates that there is no universally correct cache composition.

### Panel (d): Dynamic bottleneck shift

Run one continuous trace while changing available resources in controlled epochs:

1. bandwidth-constrained;
2. CPU-constrained;
3. bandwidth-constrained again;
4. optional mixed-pressure interval.

Plot synchronized time series for:

- throughput or p99 model-pull latency;
- cache composition: base bytes versus materialized-target bytes;
- S3 and decode utilization.

Compare:

- static hit-oriented policy;
- static base-biased policy;
- adaptive reconstruction-aware policy.

The desired result is not merely that the adaptive policy has the best average. It should visibly:

- increase base residency when S3 becomes congested;
- increase materialization residency when decode queues build;
- preserve low origin bytes in family bursts;
- preserve low decode work during repeated hot-model deployment;
- track or approach the better static policy in each phase without foreknowledge.

---

# 8. Experimental protocol for Figure 1

## 8.1 Catalog profiling before workload construction

Profile every candidate tensor and base:

- raw and reconstructed size;
- compressed delta size and \(r_x\);
- decode wall time and core-seconds;
- base fan-out;
- number of descendants co-occurring within each model manifest;
- number of models sharing each base;
- direct request role versus prerequisite-only role.

Report the selected subgraph against the full catalog distribution to demonstrate representativeness. For example, compare medians and quantiles for tensor size, delta ratio, fan-out, and tensors per model.

## 8.2 Trace generation

Freeze request identities and timestamps before any policy run.

Use an open-loop trace for paired policy comparisons. Include:

- a family-burst process for Phase A;
- a persistent hot set for Phase B;
- Zipf- or HF-weighted background pulls for Phase C;
- controlled concurrency sufficient to produce overlapping base misses;
- explicit model bundles rather than independent tensor requests.

All policies must replay exactly the same trace. A separate closed-loop experiment may measure saturation throughput.

## 8.3 Correctly coalesce base loads

Implement two levels of deduplication:

1. **Intra-request:** one base fetch per model pull, regardless of how many missed deltas need it.
2. **Inter-request single-flight:** one active origin fetch per base, with concurrent requesters joining as waiters.

Instrument:

- logical base requirements;
- physical S3 base GETs;
- number of intra-request reuses;
- number of inter-request coalesced waiters;
- bytes and latency saved by coalescing.

Run an ablation with single-flight disabled only to quantify its importance. Do not use the disabled version for policy conclusions.

## 8.4 Control S3 bandwidth and CPU capacity

Use an application-level byte token bucket or a calibrated origin proxy to control effective S3 bandwidth. This is more reproducible than relying on natural S3 variance.

Control decode capacity using:

- a fixed worker pool;
- CPU affinity;
- cgroup quotas;
- measured real decoding, not synthetic delay when avoidable.

Calibrate each resource independently:

- origin bandwidth with decoding bypassed;
- decode throughput with all inputs memory-resident.

Then create a grid of at least 6–10 S3 levels by 6–10 CPU levels for the phase diagram.

## 8.5 Cache capacity and initialization

Choose a capacity that forces a genuine tradeoff, for example:

- enough to hold only a fraction of active bases;
- or enough to hold the hot targets but not their full prerequisite set.

Report capacity as:

- absolute bytes;
- fraction of active raw-base footprint;
- fraction of active reconstructed working set.

Use both:

- controlled initial placements for causal micro-analysis; and
- cold-start online runs for end-to-end realism.

## 8.6 Metrics

Primary:

- origin bytes per completed model pull;
- physical S3 GETs per pull;
- decode core-seconds per pull;
- sustainable throughput;
- model-pull p50, p95, and p99 latency.

Explanatory:

- direct materialization hit ratio;
- prerequisite-hit ratio;
- byte hit ratio;
- cheap versus expensive target misses;
- distinct bases loaded per pull;
- number of deltas served per loaded base;
- single-flight coalescing ratio;
- cache bytes by role;
- per-ratio-quantile contribution to origin bytes and decode work.

The final item is important: the long-range compression distribution may cause a small subset of poorly compressed targets to dominate traffic.

## 8.7 Repetitions and statistical presentation

For each Figure 1 point:

- use at least three independent system runs;
- use several trace seeds where randomness is present;
- retain paired comparisons on identical traces;
- show confidence intervals for throughput and p99 latency;
- verify steady state with cache-composition and queue-length traces.

The scatter in Panel (b) should preferably aggregate multiple trace windows, while annotated points should correspond to reproducible, named scenarios.

---

# 9. Recommended Figure 1 caption

> **Figure 1: More cache hits can cost more in a reconstruction cache.** We replay a catalog-grounded workload containing heterogeneous delta ratios, shared bases, bundled model pulls, and overlapping requests. A hit-oriented allocation achieves a higher reconstructed-target hit ratio yet moves more S3 bytes because its remaining misses repeatedly require absent bases plus deltas; a base-aware allocation records fewer direct hits but converts many misses into delta-only reconstructions, with each base loaded once and shared across all dependent misses. The preferred allocation changes across the S3-bandwidth/decode-capacity plane. Under a dynamic bottleneck shift, a reconstruction-aware policy changes the cache mix between bases and materializations and tracks the best regime-specific behavior.

---

# 10. Figure-design cautions

### Keep the toy example as explanation, not evidence

The two-target example is useful for deriving intuition and an analytical crossover, but the signature figure should be measured on the heterogeneous catalog-grounded workload.

### Preserve complete model bundles

Sampling tensors independently removes the co-access structure that determines whether a base is fetched once and shared across many deltas.

### Do not use the mean compression ratio as the workload

The visible workload should preserve the empirical \(r_x\) distribution. Use the \(0.33\)–\(0.34\) mean only for prose intuition.

### Distinguish logical misses from physical origin work

Report both:

- number of delta misses and logical base requirements;
- number of physical base fetches after intra-request reuse and inter-request single-flight coalescing.

### Do not call a prerequisite lookup an ordinary target hit

Report direct materialization hits and prerequisite hits separately.

### Use fixed open-loop arrivals for paired comparisons

A closed-loop client changes the arrival trace when policy latency changes. Use closed-loop only for a separate saturation-throughput experiment.

### Avoid manufacturing the policy crossover

Choose resource levels before examining policy outcomes, use a broad two-dimensional sweep, and report all grid points. The phase boundary should emerge from measured service demand.

### Do not claim one allocation is universally optimal

The central result is that the correct cache contents depend on workload structure, compression heterogeneity, and current resource pressure.

### Do not optimize only origin bytes

A bytes-only objective will remain base-heavy when decode CPU is the actual bottleneck. Use dynamically weighted bandwidth and compute costs.


---

# 11. Broader evaluation questions suggested by Figure 1

## RQ1: How often does hit ratio mis-rank policies on the full TensorDex workload?

For every policy-capacity cell, plot:

- direct target hit ratio;
- origin GB/hour;
- decode core-seconds/hour;
- p99 model-pull latency.

Count the fraction of policy pairs for which hit-ratio ordering disagrees with origin-cost or latency ordering.

## RQ2: What cache composition is optimal in different resource regimes?

Sweep:

- S3 bandwidth;
- decode-worker count;
- cache capacity.

Plot the fraction of cache bytes occupied by:

- bases;
- reconstructed targets;
- optionally deltas.

A phase diagram can show base-heavy, mixed, and target-heavy regions.

## RQ3: Can a dynamic policy track bottleneck changes?

Use time-varying resource availability and bursty demand. Compare adaptation time, throughput loss during transitions, and eviction churn.

## RQ4: How much does bundle awareness matter?

Compare:

- independent per-target scoring;
- base fan-out scoring;
- request-bundle-aware marginal scoring.

Measure overvaluation caused by counting the same base cost repeatedly within one model pull. Also measure the gap between logical base requirements and physical S3 loads after intra-request reuse and inter-request single-flight coalescing.

## RQ5: What is the policy overhead?

Report:

- score-update time;
- metadata memory;
- number of affected descendants per update;
- heap operations;
- CPU percentage consumed by policy logic;
- lock contention under high concurrency.


## RQ6: How much does compression-ratio heterogeneity matter?

Compare:

- a policy using the global mean ratio;
- a policy using per-family mean ratios;
- a policy using exact per-tensor \(d_x\) and decode cost.

Break down errors and savings by compression-ratio quantile and tensor-size quantile. Determine whether the long tail changes only fine-grained placement or moves the resource-regime boundary itself.

---

# 12. Recommended paper-level problem statement

> A delta-compressed object store creates a reconstruction cache in which miss cost depends on prerequisite residency and tensor-specific compressibility: a target miss requires its heterogeneous delta when the base is resident, but requires the base plus delta otherwise. A single loaded base can serve several delta misses in one model pull and can be shared across overlapping in-flight pulls. Bases and materialized targets compete for capacity while saving different resources—bases reduce shared origin traffic across descendants, whereas materializations additionally eliminate decode work. Given bundled requests, finite capacity, long-range compression ratios, concurrency, and time-varying origin and compute bottlenecks, the cache must jointly choose admission and eviction to minimize bottleneck-weighted physical service demand rather than hit ratio.

---

# 13. Recommended one-sentence objective

> Show that conventional hit-rate and average-cost policies mismanage reconstruction caches because prerequisite residency, shared base loads, tensor-specific compression ratios, and changing resource bottlenecks make physical miss costs state dependent, and demonstrate that a bottleneck-aware marginal-value policy improves origin traffic, sustainable throughput, and tail latency by dynamically choosing between bases and materialized targets.

---

## Final takeaway

The strongest paper story is not that TensorDex has a DAG and therefore needs a DAG-aware score. The stronger insight is:

> **Compression changes the semantics of a cache hit.**

A reconstructed-target hit avoids both transfer and compute. A resident base may create no direct hit at all, yet one physical load of that base can eliminate repeated transfer cost across many heterogeneous descendants and concurrent requests. Because delta ratios span a long range, the correct allocation must be made at tensor and request-bundle granularity. It is determined by marginal physical service demand under the current bottleneck—not by hit ratio, average compression, static object cost, or fan-out alone.

Figure 1 should make that inversion visible before the paper introduces the full system.
