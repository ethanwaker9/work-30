import argparse
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

from ..softmax import ours as O
from ..softmax import baselines as B
from .complexity import ALL as CCOUNT

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.titlesize": 8,
    "lines.linewidth": 1.1, "lines.markersize": 3.4, "figure.dpi": 300,
    "ps.useafm": False, "pdf.fonttype": 42, "ps.fonttype": 42,
})

STYLE = {
    "ACS": ("k", "o", "-"), "Cho24": ("tab:blue", "s", "--"),
    "THOR": ("tab:red", "^", "-."), "NEXUS": ("tab:green", "v", ":"),
    "HETAL": ("tab:purple", "D", "--"), "2Quad": ("tab:orange", "x", ":"),
}
OFFSET = {"ACS-prov": (4, 4), "ACS-emp": (-24, 4), "HETAL": (-26, -2),
          "THOR": (4, 3), "Cho24": (4, 3), "2Quad": (4, -1), "NEXUS": (4, 3)}
LABEL = {"ACS": "ACS (ours)", "Cho24": "Cho et al.", "THOR": "THOR", "NEXUS": "NEXUS",
         "HETAL": "HETAL", "2Quad": "2Quad"}


def save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="eps", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    print("wrote", path)


def fig_depth_vs_range(out, n=128, alpha=16):
    spreads = np.unique(np.round(np.geomspace(8, 4096, 14)))
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.05))
    zoom = ax[0].inset_axes([0.1, 0.63, 0.44, 0.33])
    for m, fn in CCOUNT.items():
        if m in ("NEXUS", "2Quad"):
            continue
        dv, mv = [], []
        for s in spreads:
            r = fn(n, float(s), alpha)
            dv.append(r["depth"])
            mv.append(r["mult"] + r["rot"])
        c, mk, ls = STYLE[m]
        lw = 1.6 if m == "ACS" else 1.0
        ax[0].plot(spreads, dv, color=c, marker=mk, ls=ls, lw=lw, label=LABEL[m])
        ax[1].plot(spreads, mv, color=c, marker=mk, ls=ls, lw=lw, label=LABEL[m])
        if m in ("Cho24", "ACS"):
            zoom.plot(spreads, dv, color=c, marker=mk, ls=ls, lw=lw, ms=2.4)
    for a, yl in zip(ax, ["multiplicative levels", "key switches per softmax"]):
        a.set_xscale("log", base=2)
        a.set_xlabel("provable score half-spread $S$")
        a.set_ylabel(yl)
        a.grid(alpha=0.25, lw=0.4)
        a.set_ylim(bottom=0)
    ax[0].set_ylim(top=370)
    ax[1].set_ylim(top=400)
    ax[1].legend(ncol=2, frameon=False, loc="lower right", fontsize=6.5, handlelength=2.6,
                 columnspacing=1.0)
    zoom.set_xscale("log", base=2)
    zoom.set_ylim(10, 105)
    zoom.tick_params(labelsize=5, length=2, pad=1)
    zoom.set_xticks([2 ** 4, 2 ** 8, 2 ** 12])
    zoom.set_xticklabels(["$2^4$", "$2^8$", "$2^{12}$"])
    zoom.grid(alpha=0.25, lw=0.3)
    save(fig, out)


def fig_surface(out, alpha=16):
    ns = np.array([32, 64, 128, 256, 512])
    ss = np.unique(np.round(np.geomspace(8, 2048, 10)))
    Z = np.zeros((len(ns), len(ss)))
    for i, n in enumerate(ns):
        for j, s in enumerate(ss):
            Z[i, j] = O.plan(int(n), float(s), alpha)["depth"]
    X, Y = np.meshgrid(np.log2(ss), np.log2(ns))
    fig = plt.figure(figsize=(3.2, 2.0))
    ax = fig.add_subplot(111, projection="3d")
    sf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", linewidth=0.25,
                         antialiased=True)
    ax.set_xlabel("$\\log_2 S$", labelpad=-6)
    ax.set_ylabel("$\\log_2 n$", labelpad=-6)
    ax.set_zlabel("levels", labelpad=-6)
    ax.tick_params(pad=-3, labelsize=6)
    ax.view_init(elev=24, azim=-126)
    ax.set_box_aspect((1.35, 1.0, 0.62), zoom=1.28)
    fig.colorbar(sf, shrink=0.55, aspect=11, pad=0.0)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.02, top=1.0)
    save(fig, out)


def fig_bounds(out, rb):
    blocks = rb["blocks"]
    ls = np.arange(len(blocks))
    cmax = [b["logit_spread"] for b in blocks]
    cmean = [b["logit_spread_mean"] for b in blocks]
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.15))
    ax[0].semilogy(ls, cmax, "k-o", label="provable, worst head")
    ax[0].semilogy(ls, cmean, "k--s", label="provable, head average")
    if "empirical" in rb:
        ax[0].semilogy(ls, rb["empirical"], "-^", color="tab:red", label="observed on text")
    ax[0].set_xlabel("block $\\ell$")
    ax[0].set_ylabel("score half-spread")
    ax[0].legend(frameon=False)
    ax[0].grid(alpha=0.25, lw=0.4)
    rin = [b["r_in"] for b in blocks] + [rb["r_final"]]
    ax[1].semilogy(np.arange(len(rin)), rin, "k-o", label="$R_\\ell$ (residual)")
    ax[1].semilogy(ls, [b["ln1"] for b in blocks], "--s", color="tab:blue",
                   label="$\\Lambda^{(1)}_\\ell$")
    ax[1].semilogy(ls, [b["gelu_in"] for b in blocks], "-.^", color="tab:green",
                   label="$G_\\ell$")
    ax[1].set_xlabel("block $\\ell$")
    ax[1].set_ylabel("provable bound")
    ax[1].legend(frameon=False)
    ax[1].grid(alpha=0.25, lw=0.4)
    save(fig, out)


def fig_flood(out):
    b = np.geomspace(2.0 ** -30, 2.0 ** 0, 60)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for q, ls in [(1, "-"), (2 ** 10, "--"), (2 ** 20, ":")]:
        need = 128 / 2 + 0.5 * np.log2(24 * q) + np.log2(b) + 12
        ax.plot(np.log2(b), need, ls, color="k", label="$q=2^{%d}$" % int(np.log2(q)))
    ax.axhline(40, color="tab:red", lw=0.8)
    ax.text(-29, 42, "scale of our parameter set", color="tab:red", fontsize=6)
    ax.set_xlabel("$\\log_2 B_C$ (provable output error bound)")
    ax.set_ylabel("required $\\log_2\\Delta$")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, out)


def fig_micro(out, rows):
    order = ["HETAL", "THOR", "Cho24", "ACS"]
    spreads = sorted(set(r["spread"] for r in rows if "error" not in r))
    fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.0))
    fig.subplots_adjust(wspace=0.42)
    w = 0.36
    for si, s in enumerate(spreads):
        xs = np.arange(len(order)) + (si - 0.5) * w
        t = [next((r["time_s"] for r in rows if r["method"] == m and r["spread"] == s), 0)
             for m in order]
        ks = [next((r["keyswitch"] for r in rows if r["method"] == m and r["spread"] == s), 0)
              for m in order]
        bt = [next((r["levels_used"] for r in rows if r["method"] == m and r["spread"] == s), 0)
              for m in order]
        lab = "$S=%g$" % int(s + 0.5)
        ax[0].bar(xs, t, w, label=lab, color="0.3" if si == 0 else "0.7", edgecolor="k", lw=0.4)
        ax[1].bar(xs, ks, w, label=lab, color="0.3" if si == 0 else "0.7", edgecolor="k", lw=0.4)
        ax[2].bar(xs, bt, w, label=lab, color="0.3" if si == 0 else "0.7", edgecolor="k", lw=0.4)
    for a, yl in zip(ax, ["latency (s)", "key switches", "levels consumed"]):
        a.set_xticks(np.arange(len(order)))
        a.set_xticklabels([LABEL[m] for m in order], rotation=30, ha="right")
        a.set_ylabel(yl)
        a.grid(axis="y", alpha=0.25, lw=0.4)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07))
    save(fig, out)


def fig_ppl(out, acc):
    names = [k for k in acc if isinstance(acc[k], dict) and "ppl" in acc[k] and k != "exact"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    for k in names:
        d = acc[k]["depth"]
        dp = abs(acc[k]["d_ppl"]) + 1e-4
        base = k.split("-")[0]
        c, mk, _ = STYLE.get(base, ("k", "o", "-"))
        ax.scatter(d, dp, color=c, marker=mk, s=22)
        off = OFFSET.get(k, (4, 3))
        ax.annotate(k, (d, dp), fontsize=6, xytext=off, textcoords="offset points")
    ax.set_yscale("log")
    ax.set_xlabel("multiplicative levels")
    ax.set_ylabel("$|\\Delta$ perplexity$|$ on WikiText-2")
    ax.grid(alpha=0.25, lw=0.4)
    save(fig, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="../final_paper/figures")
    a = ap.parse_args()
    fig_depth_vs_range(os.path.join(a.out, "depth_range.eps"))
    fig_surface(os.path.join(a.out, "surface3d.eps"))
    fig_flood(os.path.join(a.out, "flood.eps"))
    cp = os.path.join(a.results, "range_bounds.json")
    if os.path.exists(cp):
        fig_bounds(os.path.join(a.out, "bounds.eps"), json.load(open(cp)))
    mp = os.path.join(a.results, "microbench.json")
    if os.path.exists(mp):
        fig_micro(os.path.join(a.out, "micro.eps"), json.load(open(mp)))
    ap2 = os.path.join(a.results, "gpt2_accuracy.json")
    if os.path.exists(ap2):
        fig_ppl(os.path.join(a.out, "ppl.eps"), json.load(open(ap2)))


if __name__ == "__main__":
    main()
