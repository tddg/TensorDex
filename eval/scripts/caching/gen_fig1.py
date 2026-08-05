#!/usr/bin/env python
"""Render the reconstruction-aware caching motivational figure (Figure 1).

Consumes fig1_sim.py output JSON; emits paper-style PDF + PNG.

Four panels:
  (a) anatomy — per-tensor delta-ratio ECDF (count & byte weighted) with
      base fan-in CCDF inset: miss cost is heterogeneous and shared
  (b) the inversion — direct hit ratio vs origin bytes/pull across
      policy x capacity cells (trajectories over capacity)
  (c) mechanism — origin-byte decomposition at equal capacity
  (d) regime dependence — winning policy over the S3-bandwidth x
      decode-capacity plane, from measured per-pull demand vectors
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# validated categorical trio (light surface) + recessive baseline gray
C_RECON, C_GDSF, C_BASE, C_LRU = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
POL = {
    "recon": dict(color=C_RECON, marker="o", label="reconstruction-aware"),
    "gdsf": dict(color=C_GDSF, marker="s", label="GDSF (cost-aware)"),
    "basebias": dict(color=C_BASE, marker="D", label="base-biased"),
    "lru": dict(color=C_LRU, marker="^", label="LRU (hit-oriented)"),
}

plt.rcParams.update({
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "font.family": ["DejaVu Sans"], "axes.edgecolor": "#c3c2b7",
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def panel_a(ax, anatomy):
    q = np.linspace(0, 1, 201)
    ax.plot(anatomy["ratio_q"], q, color=C_RECON, lw=1.4)
    ax.plot(anatomy["ratio_q_bytes"], q, color=C_RECON, lw=1.4, ls="--")
    ax.text(0.30, 0.26, "by tensor", color=C_RECON, fontsize=6)
    ax.text(0.44, 0.72, "by bytes", color=C_RECON, fontsize=6, style="italic")
    mw = anatomy["ratio_mean_w"]
    ax.axvline(mw, color=MUTED, lw=0.6, ls=":")
    ax.text(mw + 0.02, 0.02, f"mean {mw:.2f}", fontsize=5.5, color=MUTED)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"delta ratio  $r_x = d_x/L_x$")
    ax.set_ylabel("CDF")
    ax.set_title("(a) Miss cost is heterogeneous\nand shared", loc="left")

    fan = np.array(anatomy["fanin"], dtype=float)
    ins = ax.inset_axes([0.52, 0.13, 0.44, 0.40])
    ins.loglog(np.arange(1, len(fan) + 1), fan, color=C_GDSF, lw=1.2)
    ins.set_xlabel("base rank", fontsize=5, labelpad=1)
    ins.set_ylabel("fan-in", fontsize=5, labelpad=1)
    ins.tick_params(labelsize=5, pad=1)
    ins.text(0.05, 0.13, f"max {int(fan.max()):,}\ndeltas/base",
             transform=ins.transAxes, fontsize=5, color=INK2)
    for a in (ax, ins):
        a.grid(True, color=GRID, lw=0.4)
        a.set_axisbelow(True)


def panel_b(ax, results, annotate_cap):
    by_pol = {}
    for r in results:
        by_pol.setdefault(r["policy"], []).append(r)
    for pol, rows in by_pol.items():
        rows.sort(key=lambda r: r["cap_gb"])
        x = [100 * r["hit_ratio"] for r in rows]
        y = [r["origin_gb_per_pull"] for r in rows]
        st = POL[pol]
        ax.plot(x, y, color=st["color"], lw=0.9, alpha=0.8, zorder=2)
        ax.scatter(x, y, s=[5 + 5 * i for i in range(len(rows))],
                   color=st["color"], marker=st["marker"], zorder=3,
                   edgecolors="white", linewidths=0.4)
        if pol == "lru":
            for r, xi, yi in zip(rows, x, y):
                ax.annotate(f"{r['cap_gb']:.0f}", (xi, yi),
                            textcoords="offset points", xytext=(-1, 5),
                            fontsize=5.0, color=MUTED, ha="right")
        if pol == "recon":
            for r, xi, yi in zip(rows, x, y):
                ax.annotate(f"{r['cap_gb']:.0f} GB", (xi, yi),
                            textcoords="offset points", xytext=(3, -8),
                            fontsize=5.0, color=MUTED)
    a = next(r for r in results
             if r["policy"] == "gdsf" and r["cap_gb"] == annotate_cap)
    b = next(r for r in results
             if r["policy"] == "recon" and r["cap_gb"] == annotate_cap)
    ax.annotate("", xy=(100 * a["hit_ratio"], a["origin_gb_per_pull"]),
                xytext=(100 * b["hit_ratio"], b["origin_gb_per_pull"]),
                arrowprops=dict(arrowstyle="->", color=INK, lw=0.8))
    hx = 100 * (a["hit_ratio"] + b["hit_ratio"]) / 2
    hy = (a["origin_gb_per_pull"] + b["origin_gb_per_pull"]) / 2
    ax.text(hx - 1, hy + 0.75,
            f"same {annotate_cap:.0f} GB cache:\n"
            f"{a['hit_ratio']/b['hit_ratio']:.1f}$\\times$ the hits,\n"
            f"{a['origin_gb_per_pull']/b['origin_gb_per_pull']:.1f}$\\times$"
            " the S3 traffic",
            fontsize=5.8, color=INK, ha="left")
    ax.set_xlabel("direct hit ratio (%)")
    ax.set_ylabel("origin bytes / model pull (GB)")
    ax.set_title("(b) More hits can cost more\n(capacities 25–400 GB labeled)",
                 loc="left")
    ax.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)


def panel_c(ax, results, cap):
    order = ["lru", "gdsf", "basebias", "recon"]
    rows = {r["policy"]: r for r in results if r["cap_gb"] == cap}
    comps = [
        ("delta_only_gb", "#86b6ef", "delta (base resident)"),
        ("delta_with_base_gb", "#1c5cab", "delta (base absent)"),
        ("base_gb", C_GDSF, "base refetch"),
        ("raw_gb", "#c3c2b7", "raw"),
    ]
    xs = np.arange(len(order))
    bottom = np.zeros(len(order))
    for key, color, label in comps:
        v = np.array([rows[p][key] for p in order])
        ax.bar(xs, v, 0.62, bottom=bottom, color=color, label=label,
               edgecolor="white", linewidth=0.6)
        bottom += v
    for i, p in enumerate(order):
        ax.text(xs[i], bottom[i] + 0.25, f"{bottom[i]:.1f}", ha="center",
                fontsize=5.8, color=INK2)
    ax.set_xticks(xs)
    ax.set_xticklabels(["LRU", "GDSF", "base-\nbiased", "recon-\naware"],
                       fontsize=6)
    ax.set_ylabel(f"origin GB / pull  ({cap:.0f} GB cache)")
    ax.set_title("(c) Base refetch is the\navoidable traffic", loc="left")
    ax.legend(frameon=False, fontsize=5.2, loc="upper right",
              handlelength=1.0, borderaxespad=0.1)
    ax.grid(True, axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(bottom) * 1.22)


def panel_e(ax, results, cap, bw_gbps, cores):
    """Model-pull latency at the operating point: serial fetch+decode model.

    fetch_t = origin bytes / node origin bandwidth; decode_t = decode
    core-seconds / cores. Serial sum is an upper bound (perfect pipelining
    approaches max of the two); quantiles over per-pull distributions.
    """
    order = ["lru", "gdsf", "basebias", "recon"]
    rows = {r["policy"]: r for r in results if r["cap_gb"] == cap}
    xs = np.arange(len(order))
    C_FETCH, C_DEC = "#6da7ec", "#eda100"
    t95_max = 0.0
    for i, p in enumerate(order):
        og = np.array(rows[p]["pull_origin_gb"])
        ds = np.array(rows[p]["pull_decode_s"])
        fetch_t = og / bw_gbps
        dec_t = ds / cores
        tot = fetch_t + dec_t
        f50, d50 = np.median(fetch_t), np.median(dec_t)
        t95 = np.quantile(tot, 0.95)
        ax.bar(i, f50, 0.6, color=C_FETCH, edgecolor="white", linewidth=0.6,
               label="origin fetch (p50)" if i == 0 else None)
        ax.bar(i, d50, 0.6, bottom=f50, color=C_DEC, edgecolor="white",
               linewidth=0.6, label="decode (p50)" if i == 0 else None)
        ax.plot([i, i], [f50 + d50, t95], color=INK2, lw=0.7, zorder=4)
        ax.plot(i, t95, marker="v", color=INK2, ms=3, zorder=4,
                label="p95 total" if i == 0 else None)
        ax.text(i + 0.09, t95, f"{t95:.0f}", ha="left", va="center",
                fontsize=5.4, color=INK2)
        ax.text(i - 0.09, f50 + d50 + 0.2, f"{f50 + d50:.1f}", ha="right",
                fontsize=5.8, color=INK)
        t95_max = max(t95_max, t95)
    ax.set_xticks(xs)
    ax.set_xticklabels(["LRU", "GDSF", "base-\nbiased", "recon-\naware"],
                       fontsize=6)
    ax.set_ylabel(f"model-pull latency (s)  ({cap:.0f} GB cache)")
    ax.set_title(f"(d) Latency at the operating point\n"
                 f"({bw_gbps:g} GB/s origin, {cores} cores)", loc="left")
    ax.legend(frameon=False, fontsize=5.2, loc="upper right",
              handlelength=1.0, borderaxespad=0.1)
    ax.grid(True, axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_ylim(0, t95_max * 1.32)


def panel_d(ax, results, cap, op_point):
    rows = [r for r in results if r["cap_gb"] == cap]
    pols = [r["policy"] for r in rows]
    bytes_pp = np.array([r["origin_gb_per_pull"] for r in rows])   # GB
    cpu_pp = np.array([r["decode_s_per_pull"] for r in rows])      # core-s
    B = np.logspace(np.log10(0.25), np.log10(16), 220)             # GB/s
    D = np.logspace(np.log10(1), np.log10(256), 220)               # cores
    BB, DD = np.meshgrid(B, D)
    mu = np.minimum(BB[None] / bytes_pp[:, None, None],
                    DD[None] / cpu_pp[:, None, None])
    win = mu.argmax(axis=0)
    present = sorted(set(win.ravel()))
    cmap = ListedColormap([POL[pols[i]]["color"] for i in present])
    remap = np.zeros_like(win)
    for k, i in enumerate(present):
        remap[win == i] = k
    ax.pcolormesh(BB, DD, remap, cmap=cmap, alpha=0.42, shading="auto",
                  rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(*op_point, marker="*", color=INK, ms=8, zorder=5)
    ax.text(op_point[0] * 1.15, op_point[1] * 0.92, "this paper's\ncache node",
            fontsize=5.5, color=INK, va="top")
    # direct region labels at region centroids (log space)
    for k, i in enumerate(present):
        mask = remap == k
        if mask.sum() < 50:
            continue
        cx = np.exp(np.log(BB[mask]).mean())
        cy = np.exp(np.log(DD[mask]).mean())
        ax.text(cx, cy, POL[pols[i]]["label"].split(" (")[0].replace(
            "reconstruction-aware", "reconstruction-\naware"),
            ha="center", va="center", fontsize=6, color=INK, weight="bold")
    ax.set_xlabel("origin (S3) bandwidth (GB/s)")
    ax.set_ylabel("decode capacity (cores)")
    ax.set_title("(e) The best policy depends\non the bottleneck", loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--cap", type=float, default=100.0,
                    help="headline equal-capacity cell (GB)")
    args = ap.parse_args()
    data = json.load(open(args.results))
    results, anatomy = data["results"], data["anatomy"]

    fig, axes = plt.subplots(1, 5, figsize=(16.4, 2.9))
    fig.subplots_adjust(left=0.04, right=0.995, top=0.82, bottom=0.17,
                        wspace=0.36)
    panel_a(axes[0], anatomy)
    panel_b(axes[1], results, args.cap)
    panel_c(axes[2], results, args.cap)
    panel_e(axes[3], results, args.cap, bw_gbps=1.7, cores=32)
    panel_d(axes[4], results, args.cap, op_point=(1.7, 32))

    handles = [Line2D([], [], color=POL[p]["color"], marker=POL[p]["marker"],
                      ls="-", lw=1, ms=4, label=POL[p]["label"])
               for p in ("lru", "gdsf", "basebias", "recon")]
    axes[1].legend(handles=handles, frameon=False, fontsize=5.4,
                   loc="lower left", handlelength=1.4, borderaxespad=0.2)

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out_prefix}.{ext}", dpi=300,
                    facecolor="white", bbox_inches="tight")
        print("wrote", f"{args.out_prefix}.{ext}")


if __name__ == "__main__":
    main()
