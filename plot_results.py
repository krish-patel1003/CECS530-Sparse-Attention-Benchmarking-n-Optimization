"""Make every graph from results/*.csv  ->  figures/*.png

    python plot_results.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sparse_attention import patterns as P  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": MUTED,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2.2, "lines.markersize": 7})

SEQ = {"dense": ("Dense (standard)", ORANGE, "s"), "library (PyTorch SDPA)": ("PyTorch built-in (reference)", GREEN, "^"),
       "local (±4 blocks)": ("Block-sparse: local window", BLUE, "o"),
       "longformer (±3 + 1 global)": ("Block-sparse: Longformer", VIOLET, "D"),
       "random 12.5%": ("Block-sparse: random 12.5%", DGREEN, "v")}
PAT = {"local": (BLUE, "o"), "strided": (YELLOW, "s"), "longformer": (VIOLET, "D"), "random": (DGREEN, "v")}


def load(name):
    p = RES / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=170)
    plt.close(fig)
    print("  figures/" + name + ".png")


def log2x(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))


def plain_logy(ax, unit=""):
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}{unit}" if v >= 1 else f"{v:g}{unit}"))


def seq_lines(ax, df, ycol, key_fn):
    for key, (lab, c, mk) in SEQ.items():
        d = df[key_fn(df) == key].sort_values("N")
        if len(d):
            ax.plot(d.N, d[ycol], color=c, marker=mk, label=lab, markeredgecolor=SURF)


def seq_key(df):
    return np.where(df.impl == "block-sparse", df.pattern, df.impl)


# ------------------------------------------------------------------ figures
def fig_seqlen(suffix="", tag="CPU"):
    df = load("seqlen" + suffix)
    if df is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    seq_lines(axs[0], df, "time_ms", seq_key)
    log2x(axs[0])
    plain_logy(axs[0])
    axs[0].set(title=f"(a) Time per forward pass vs sequence length ({tag})", xlabel="sequence length N (tokens)", ylabel="milliseconds (log scale)")
    axs[0].legend(fontsize=8.5)
    d = df[df.impl == "block-sparse"]
    for key in ["local (±4 blocks)", "longformer (±3 + 1 global)", "random 12.5%"]:
        lab, c, mk = SEQ[key]
        dd = d[d.pattern == key].dropna(subset=["speedup_vs_dense"]).sort_values("N")
        axs[1].plot(dd.N, dd.speedup_vs_dense, color=c, marker=mk, label=lab, markeredgecolor=SURF)
        if len(dd):
            axs[1].annotate(f"{dd.speedup_vs_dense.iloc[-1]:.1f}×", (dd.N.iloc[-1], dd.speedup_vs_dense.iloc[-1]),
                            xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    axs[1].axhline(1, color=MUTED, ls=":", lw=1.3)
    log2x(axs[1])
    axs[1].set(title=f"(b) Speed-up of block-sparse over dense ({tag})", xlabel="sequence length N (tokens)", ylabel="speed-up (×)")
    axs[1].legend(fontsize=8.5)
    save(fig, "fig_seqlen" + suffix)


def fig_flops_throughput():
    df = load("seqlen")
    if df is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
    seq_lines(axs[0], df, "tokens_per_s", seq_key)
    log2x(axs[0])
    plain_logy(axs[0])
    axs[0].set(title="(a) Throughput (CPU)", xlabel="sequence length N (tokens)", ylabel="tokens per second (log scale)")
    axs[0].legend(fontsize=8.5)
    d = df[df.impl == "library (PyTorch SDPA)"].sort_values("N")
    axs[1].plot(d.N, d.flops / 1e9, color=ORANGE, marker="s", label="dense: 4·H·N²·D")
    for key in ["local (±4 blocks)", "longformer (±3 + 1 global)", "random 12.5%"]:
        lab, c, mk = SEQ[key]
        dd = df[(df.impl == "block-sparse") & (df.pattern == key)].sort_values("N")
        axs[1].plot(dd.N, dd.flops / 1e9, color=c, marker=mk, label=lab, markeredgecolor=SURF)
    log2x(axs[1])
    plain_logy(axs[1])
    axs[1].set(title="(b) Computation per forward pass", xlabel="sequence length N (tokens)", ylabel="billions of operations (GFLOP, log)")
    axs[1].legend(fontsize=8.5)
    save(fig, "fig_flops_throughput")


def fig_sparsity(suffix="", tag="CPU"):
    df = load("sparsity" + suffix)
    if df is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax = axs[0]
    d = df[(df.impl == "block-sparse") & (df.pattern == "random")].sort_values("density")
    ax.plot(d.density, d.ideal_speedup, color=MUTED, ls=":", lw=1.6, label="ideal = 1 / density")
    for name, (c, mk) in PAT.items():
        d = df[(df.impl == "block-sparse") & (df.pattern == name)].sort_values("density")
        ax.plot(d.density, d.speedup_vs_dense, color=c, marker=mk, label=f"block-sparse: {name}", markeredgecolor=SURF)
    d = df[df.impl == "masked dense"].sort_values("density")
    ax.plot(d.density, d.speedup_vs_dense, color=PINK, ls="--", marker="v", label="masked dense (compute all, then mask)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:g}%"))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set(title=f"(a) Speed-up vs density (N = 4,096, {tag})", xlabel="fraction of blocks computed  →  sparser",
           ylabel="speed-up over dense (log scale)")
    ax.legend(fontsize=8)
    ax = axs[1]
    for name, (c, mk) in PAT.items():
        d = df[(df.impl == "block-sparse") & (df.pattern == name)].sort_values("density")
        ax.plot(d.density, d.percent_of_ideal, color=c, marker=mk, label=name, markeredgecolor=SURF)
    ax.axhline(100, color=MUTED, ls=":", lw=1.3)
    ax.set_xscale("log", base=2)
    ax.invert_xaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:g}%"))
    ax.set(title=f"(b) How close to the ideal speed-up ({tag})", xlabel="fraction of blocks computed  →  sparser",
           ylabel="% of ideal speed-up", ylim=(0, 130))
    ax.legend(fontsize=8.5)
    save(fig, "fig_sparsity" + suffix)


def fig_patterns():
    df = load("sparsity")
    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.15])
    for i, name in enumerate(["dense", "local", "strided", "longformer", "random"]):
        ax = fig.add_subplot(gs[0, i])
        m = P.make_pattern(name, 1024, 64, target_density=0.25)
        ax.imshow(m.astype(int), cmap=matplotlib.colors.ListedColormap(["#ecebe4", BLUE]), vmin=0, vmax=1)
        ax.set_title(name, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_xlabel(f"{P.density(m)*100:.0f}% of blocks", fontsize=9, color=MUTED)
    if df is not None:
        ax = fig.add_subplot(gs[1, :])
        d = df[(df.impl == "block-sparse") & (df.target == 0.125)].set_index("pattern")
        names = ["local", "strided", "longformer", "random"]
        dense_ms = df[df.impl == "dense"].time_ms.iloc[0]
        vals = [dense_ms] + [d.loc[n, "time_ms"] for n in names]
        bars = ax.bar(["dense"] + names, vals, color=[ORANGE] + [PAT[n][0] for n in names], width=0.55)
        for b, n in zip(bars, ["dense"] + names):
            txt = f"{b.get_height():.0f} ms" if n == "dense" else \
                f"{b.get_height():.1f} ms · {d.loc[n, 'speedup_vs_dense']:.1f}× ({d.loc[n, 'density']*100:.1f}% blocks)"
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02, txt, ha="center", va="bottom", fontsize=9)
        ax.set(ylabel="milliseconds", title="Time at ≈12.5% density, N = 4,096 (CPU)", ylim=(0, max(vals) * 1.25))
        ax.grid(axis="x", visible=False)
    fig.suptitle("Sparsity patterns (top: which 64×64 blocks are computed, shown at 25% density)", fontsize=12, fontweight="bold")
    save(fig, "fig_patterns")


def fig_memory():
    df = load("memory")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    st = {"dense": ("Dense (standard)", ORANGE, "s"), "masked_dense": ("Masked dense", PINK, "v"),
          "library": ("PyTorch built-in (reference)", GREEN, "^"), "v2_views": ("Block-sparse (ours), local window", BLUE, "o")}
    for impl, (lab, c, mk) in st.items():
        d = df[df.impl == impl].sort_values("N")
        ax.plot(d.N, d.extra_mb.clip(lower=0.5), color=c, marker=mk, label=lab, markeredgecolor=SURF)
    Ns = np.array(sorted(df.N.unique()))
    ax.plot(Ns, 2 * 8 * Ns.astype(float) ** 2 * 4 / 2**20, color=MUTED, ls=":", lw=1.5, label="prediction: two N×N matrices (scores + weights) = 2·8·N²·4 bytes")
    log2x(ax)
    plain_logy(ax, " MB")
    ax.set(title="Measured peak extra memory of one attention call (CPU)", xlabel="sequence length N (tokens)", ylabel="extra memory (log scale)")
    ax.legend(fontsize=8.5)
    save(fig, "fig_memory")


def fig_ablation():
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, suffix, tag in [(axs[0], "", "CPU"), (axs[1], "_mps", "Apple GPU")]:
        df = load("ablation" + suffix)
        if df is None:
            ax.set_visible(False)
            continue
        d = df[df.N == 4096]
        impls = [("masked dense", PINK, "masked dense"), ("v1_gather", VIOLET, "v1: gather (copy blocks)"),
                 ("v2_views", BLUE, "v2: views (final)"), ("padded_batched", YELLOW, "padded batch (rejected)")]
        pats = ["local", "longformer", "random"]
        x = np.arange(len(pats))
        w = 0.2
        for i, (impl, c, lab) in enumerate(impls):
            vals = [d[(d.impl == impl) & (d.pattern == p)].time_ms.min() for p in pats]
            bars = ax.bar(x + (i - 1.5) * w, vals, w, color=c, label=lab)
            for b, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.0f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x, pats)
        ax.set(title=f"Versions compared, N = 4,096, ≈12.5% density ({tag})", ylabel="milliseconds (lower is better)")
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8)
    save(fig, "fig_ablation")


def fig_blocksize():
    df = load("blocksize")
    if df is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    lab = df.block_size.astype(str)
    axs[0].bar(lab, df.useful_fraction * 100, color=BLUE, width=0.55)
    for x, v in zip(lab, df.useful_fraction * 100):
        axs[0].text(x, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    axs[0].set(title="(a) Useful work inside computed blocks", xlabel="block size (tokens)", ylabel="% of computed entries actually needed", ylim=(0, 110))
    axs[1].bar(lab, df.time_ms, color=VIOLET, width=0.55)
    for x, v in zip(lab, df.time_ms):
        axs[1].text(x, v * 1.02, f"{v:.1f} ms", ha="center", fontsize=9)
    axs[1].set(title="(b) Time (±256-token window, N = 4,096, CPU)", xlabel="block size (tokens)", ylabel="milliseconds", ylim=(0, df.time_ms.max() * 1.2))
    for ax in axs:
        ax.grid(axis="x", visible=False)
    save(fig, "fig_blocksize")


def fig_threads():
    df = load("threads")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.axvspan(4.5, 10.5, color=GRID, alpha=0.6, lw=0)
    ax.text(7.5, 0.4, "efficiency cores", ha="center", color=MUTED, fontsize=9)
    for impl, c, mk in [("dense", ORANGE, "s"), ("block-sparse", BLUE, "o")]:
        d = df[df.impl == impl].sort_values("threads")
        ax.plot(d.threads, d.speedup_vs_1_thread, color=c, marker=mk, label=impl, markeredgecolor=SURF)
    ax.plot([1, 10], [1, 10], color=MUTED, ls=":", lw=1.3, label="perfect scaling")
    ax.set(title="CPU threads (N = 4,096): speed-up vs 1 thread", xlabel="threads", ylabel="speed-up (×)",
           xticks=[1, 2, 4, 6, 8, 10], ylim=(0, 10.5))
    ax.legend(fontsize=9)
    save(fig, "fig_threads")


def fig_cpu_vs_gpu():
    c, g = load("seqlen"), load("seqlen_mps")
    if c is None or g is None:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for df, ls, dev in [(c, "-", "CPU"), (g, "--", "GPU")]:
        for key, col, mk in [("dense", ORANGE, "s"), ("local (±4 blocks)", BLUE, "o")]:
            d = df[seq_key(df) == key].sort_values("N")
            ax.plot(d.N, d.time_ms, color=col, ls=ls, marker=mk, label=f"{'dense' if key == 'dense' else 'block-sparse local'} · {dev}", markeredgecolor=SURF)
    log2x(ax)
    plain_logy(ax)
    ax.set(title="Same code on CPU and Apple GPU", xlabel="sequence length N (tokens)", ylabel="milliseconds (log scale)")
    ax.legend(fontsize=9)
    save(fig, "fig_cpu_vs_gpu")


def render_logs():
    for p in sorted((RES / "logs").glob("*.txt")):
        lines = [ln for ln in p.read_text().rstrip().splitlines() if "done" not in ln][:46]
        width = max(len(ln) for ln in lines)
        h = 0.17 * len(lines) + 0.55
        fig = plt.figure(figsize=(max(6, 0.068 * width + 0.6), h))
        fig.patch.set_facecolor("#1e1e1e")
        fig.text(0.01, 0.985, f"$ python run_experiments.py --exp {p.stem.replace('_mps', ' --device mps')}",
                 color="#9cdcfe", family="monospace", fontsize=8.5, va="top")
        fig.text(0.01, 0.985 - 0.27 / h, "\n".join(lines), color="#e6e6e6", family="monospace", fontsize=8.5, va="top", linespacing=1.25)
        fig.savefig(FIG / f"term_{p.stem}.png", dpi=150, facecolor="#1e1e1e")
        plt.close(fig)
        print("  figures/term_" + p.stem + ".png")


if __name__ == "__main__":
    fig_seqlen()
    fig_seqlen("_mps", "Apple GPU")
    fig_flops_throughput()
    fig_sparsity()
    fig_sparsity("_mps", "Apple GPU")
    fig_patterns()
    fig_memory()
    fig_ablation()
    fig_blocksize()
    fig_threads()
    fig_cpu_vs_gpu()
    render_logs()
