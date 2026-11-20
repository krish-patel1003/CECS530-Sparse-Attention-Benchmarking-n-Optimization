"""Visuals for the project walkthrough video.

Every visual is a function  f(fig, ax, p)  that draws one 1920x1080 frame.
  ax : a full-frame axes in pixel coordinates (x right, y DOWN, origin top-left)
  p  : progress through the beat, 0 -> 1 (animations finish by p ~ 0.85 and then hold)
VIS maps a beat's key to (function, animated?, dark background?).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sparse_attention import patterns as PAT  # noqa: E402

RES, FIG = ROOT / "results", ROOT / "figures"
W, H = 1920, 1080

# ------------------------------------------------------------------ palette & fonts
INK, BODY, MUTED = "#14213D", "#3D4656", "#6B7280"
PAPER, PAPER2, LINE = "#F7F6F2", "#ECEBE4", "#D6D4CC"
BLUE, ORANGE, GREEN, YELLOW, PINK, DGREEN, VIOLET, RED = (
    "#2A78D6", "#E0612C", "#1BAF7A", "#EDA100", "#E87BA4", "#008300", "#6250D6", "#E34948")
NAVY, NAVY2, SOFT = "#14213D", "#1F3157", "#C9D2E3"
OINK = "#A8441A"
FONT = ["Avenir Next", "Helvetica Neue", "DejaVu Sans"]
MONO = ["Menlo", "DejaVu Sans Mono"]
plt.rcParams.update({"font.family": FONT, "mathtext.fontset": "dejavusans", "axes.unicode_minus": False})

# ------------------------------------------------------------------ data
seq = pd.read_csv(RES / "seqlen.csv")
seq_g = pd.read_csv(RES / "seqlen_mps.csv")
spar = pd.read_csv(RES / "sparsity.csv")
spar_g = pd.read_csv(RES / "sparsity_mps.csv")
mem = pd.read_csv(RES / "memory.csv")
abl = pd.read_csv(RES / "ablation.csv")
abl_g = pd.read_csv(RES / "ablation_mps.csv")
bsz = pd.read_csv(RES / "blocksize.csv")
thr = pd.read_csv(RES / "threads.csv")
_img_cache: dict = {}


# ================================================================== helpers
def ease(p, a=0.0, b=1.0):
    if b <= a:
        return 1.0 if p >= a else 0.0
    x = min(max((p - a) / (b - a), 0.0), 1.0)
    return x * x * (3 - 2 * x)


def T(ax, x, y, s, size=30, color=INK, weight="normal", ha="left", va="top", alpha=1.0, family=None, **kw):
    if alpha <= 0.001:
        return None
    return ax.text(x, y, s, fontsize=size * 0.75, color=color, fontweight=weight, ha=ha, va=va,
                   alpha=alpha, family=family or FONT, **kw)


def R(ax, x, y, w, h, fc, ec="none", lw=0, alpha=1.0, r=0, z=1):
    if alpha <= 0.001:
        return None
    if r:
        patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                               fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    else:
        patch = Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z)
    ax.add_patch(patch)
    return patch


def arrow(ax, x0, y0, x1, y1, color=MUTED, lw=3, alpha=1.0, rad=0.0, head=22):
    if alpha <= 0.001:
        return
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=f"-|>,head_length={head/18},head_width={head/30}",
                                 mutation_scale=18, color=color, lw=lw, alpha=alpha,
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))


def header(ax, eyebrow, title, p=1.0, dark=False):
    a = ease(p, 0, 0.12)
    T(ax, 128, 92, eyebrow.upper(), 22, "#F2A27E" if dark else OINK, "bold", alpha=a)
    T(ax, 128, 128, title, 54, PAPER if dark else INK, "bold", alpha=a)


def pill(ax, x, y, text, fc=BLUE, color=PAPER, size=24, alpha=1.0, pad=18, h=None):
    w = len(text) * size * 0.56 + 2 * pad
    hh = h or size * 1.9
    R(ax, x, y, w, hh, fc, alpha=alpha, r=hh / 2, z=3)
    T(ax, x + w / 2, y + hh / 2, text, size, color, "bold", ha="center", va="center", alpha=alpha, zorder=4)
    return w


def card(ax, x, y, w, h, title, body, accent=BLUE, alpha=1.0, tsize=30, bsize=24, dark=False):
    if alpha <= 0.001:
        return
    R(ax, x, y, w, h, NAVY2 if dark else "#FFFFFF", ec=LINE if not dark else NAVY2, lw=1.5, alpha=alpha, r=18)
    R(ax, x, y, 10, h, accent, alpha=alpha, r=4)
    T(ax, x + 36, y + 30, title, tsize, PAPER if dark else INK, "bold", alpha=alpha, wrap=False)
    T(ax, x + 36, y + 30 + tsize * 1.55, body, bsize, SOFT if dark else BODY, alpha=alpha, linespacing=1.45)


def inset(fig, x, y, w, h):
    """Axes placed by pixel box (top-left origin)."""
    ax = fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H])
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(1.5)
    ax.tick_params(colors=BODY, labelsize=17, length=6, width=1.2)
    ax.grid(True, color=LINE, lw=1, alpha=0.8)
    ax.set_axisbelow(True)
    return ax


def axlabels(ax, xl, yl):
    ax.set_xlabel(xl, fontsize=20, color=INK, labelpad=10)
    ax.set_ylabel(yl, fontsize=20, color=INK, labelpad=10)


def logfmt(ax, axis="x", base=2):
    f = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(f)
    if axis == "x":
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))


def partial_line(ax, xs, ys, frac, logx=True, logy=True, **kw):
    """Draw a polyline up to `frac` of its points (interpolating the last segment)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = len(xs)
    if n == 0 or frac <= 0:
        return
    t = frac * (n - 1)
    k = int(math.floor(t))
    px, py = list(xs[: k + 1]), list(ys[: k + 1])
    if k < n - 1:
        u = t - k
        fx = (lambda a, b: a * (b / a) ** u) if logx else (lambda a, b: a + (b - a) * u)
        fy = (lambda a, b: a * (b / a) ** u) if logy else (lambda a, b: a + (b - a) * u)
        px.append(fx(xs[k], xs[k + 1]))
        py.append(fy(ys[k], ys[k + 1]))
    mk = kw.pop("marker", "o")
    ax.plot(px, py, **kw, solid_capstyle="round")
    ax.plot(xs[: k + 1], ys[: k + 1], ls="none", marker=mk, ms=kw.get("ms", 11), color=kw.get("color"),
            mec=PAPER, mew=2, zorder=kw.get("zorder", 3))


def image(fig, path, x, y, w, h, zoom=1.0):
    if path not in _img_cache:
        _img_cache[path] = plt.imread(path)
    im = _img_cache[path]
    ih, iw = im.shape[:2]
    s = min(w / iw, h / ih)
    dw, dh = iw * s, ih * s
    ax = fig.add_axes([(x + (w - dw) / 2) / W, 1 - (y + (h - dh) / 2 + dh) / H, dw / W, dh / H])
    ax.imshow(im, interpolation="lanczos")
    if zoom > 1.0:
        cx, cy = iw / 2, ih / 2
        ax.set_xlim(cx - iw / 2 / zoom, cx + iw / 2 / zoom)
        ax.set_ylim(cy + ih / 2 / zoom, cy - ih / 2 / zoom)
    ax.axis("off")
    return ax


def grid_tiles(ax, x, y, mask, cell, on=BLUE, off="#E4E2DA", gap=3, alpha=1.0, colors=None):
    n = mask.shape[0]
    for i in range(n):
        for j in range(mask.shape[1]):
            c = (colors[i][j] if colors is not None else (on if mask[i, j] else off))
            R(ax, x + j * cell, y + i * cell, cell - gap, cell - gap, c, alpha=alpha, r=min(6, cell * 0.12))


def chrome(ax, info, dark=False):
    """Bottom bar: chapter title + whole-video progress with chapter ticks."""
    col = SOFT if dark else MUTED
    T(ax, 128, 1012, f"{info['chapter_no']:02d}  ·  {info['chapter']}", 20, col, "bold")
    T(ax, 1792, 1012, "Sparse Attention Benchmarking", 20, col, ha="right")
    R(ax, 128, 1050, 1664, 5, "#3A4A6B" if dark else LINE, r=2)
    R(ax, 128, 1050, 1664 * info["progress"], 5, ORANGE, r=2)
    for t in info["ticks"]:
        R(ax, 128 + 1664 * t - 1, 1046, 3, 13, col, alpha=0.6)


# ================================================================== 1. introduction
def v_title(fig, ax, p, end=False):
    nb = 18
    m = PAT.longformer(nb, 2, 1)
    for i in range(nb):
        for j in range(nb):
            a = ease(p, 0.05 + 0.25 * (i + j) / (2 * nb), 0.3 + 0.25 * (i + j) / (2 * nb))
            if m[i, j]:
                R(ax, 1190 + j * 33, 170 + i * 33, 29, 29, BLUE if (i + j) % 7 else ORANGE, alpha=0.85 * a, r=5)
            else:
                R(ax, 1190 + j * 33, 170 + i * 33, 29, 29, NAVY2, alpha=a, r=5)
    a1, a2, a3 = ease(p, 0, .15), ease(p, .1, .3), ease(p, .25, .45)
    if end:
        T(ax, 128, 330, "Thanks for\nwatching", 104, PAPER, "bold", alpha=a1, linespacing=1.05)
        T(ax, 128, 600, "Code, results, report and study guide:\nsparse_attention_benchmark/", 30, SOFT, alpha=a2, linespacing=1.5)
    else:
        T(ax, 128, 220, "COMPUTER ARCHITECTURE · TERM PROJECT", 24, "#F2A27E", "bold", alpha=a1)
        T(ax, 128, 270, "Sparse Attention\nBenchmarking and\nOptimization", 92, PAPER, "bold", alpha=a2, linespacing=1.05)
        T(ax, 128, 640, "A plain-language walkthrough: the problem, our kernel,\nthe measurements and every graph", 32, SOFT, alpha=a3, linespacing=1.45)
    R(ax, 128, 800, 200, 10, ORANGE, alpha=a3)
    T(ax, 128, 835, "Krish Patel  ·  Aarsh Sanjay Sharma", 30, PAPER, "bold", alpha=a3)
    T(ax, 128, 885, "PyTorch · Apple M4 laptop (CPU and GPU)", 24, SOFT, alpha=a3)


def v_end(fig, ax, p):
    v_title(fig, ax, p, end=True)


def v_roadmap(fig, ax, p):
    header(ax, "Roadmap", "Four parts", p)
    items = [("1", "The problem", "What attention computes\nand why it gets\nexpensive"),
             ("2", "Our kernel", "Block sparsity\nthe 5-line kernel\nhow we improved it"),
             ("3", "The results", "Benchmark framework\nevery experiment\nevery graph"),
             ("4", "The lessons", "Why not the ideal\nconclusions")]
    for i, (n, t, b) in enumerate(items):
        a = ease(p, 0.08 + i * 0.17, 0.2 + i * 0.17)
        x = 128 + i * 420
        R(ax, x, 330, 392, 470, "#FFFFFF", ec=LINE, lw=1.5, alpha=a, r=22)
        T(ax, x + 36, 370, n, 110, [BLUE, ORANGE, GREEN, VIOLET][i], "bold", alpha=a)
        T(ax, x + 36, 540, t, 36, INK, "bold", alpha=a)
        T(ax, x + 36, 610, b, 26, BODY, alpha=a, linespacing=1.5)


# ================================================================== 2. attention
TOKENS = ["The", "cat", "sat", "because", "it", "was", "tired"]
WEIGHTS = [0.03, 0.70, 0.08, 0.05, 0.0, 0.06, 0.08]


def _token_row(ax, y, p_hi, size=40):
    xs, x = [], 190
    for t in TOKENS:
        w = len(t) * size * 0.62 + 60
        xs.append((x, w))
        hi = t == "it"
        R(ax, x, y, w, 96, ORANGE if (hi and p_hi > 0) else "#FFFFFF", ec=LINE, lw=2, r=16,
          alpha=1.0)
        if hi and p_hi > 0:
            R(ax, x, y, w, 96, ORANGE, alpha=p_hi, r=16)
        T(ax, x + w / 2, y + 48, t, size, PAPER if (hi and p_hi > 0.5) else INK, "bold", ha="center", va="center")
        x += w + 34
    return xs


def v_attn_sentence(fig, ax, p):
    header(ax, "What attention computes", "Every token looks at every other token", p)
    xs = _token_row(ax, 560, ease(p, 0.3, 0.45))
    ix, iw = xs[4]
    a = ease(p, 0.45, 0.85)
    for k, (x, w) in enumerate(xs):
        if k == 4:
            continue
        wgt = WEIGHTS[k]
        arrow(ax, ix + iw / 2, 560, x + w / 2, 560, color=ORANGE if k == 1 else MUTED, lw=2 + 14 * wgt,
              alpha=a * (0.35 + 0.65 * (wgt > 0.5)), rad=0.45 if x < ix else -0.45, head=16 + 14 * wgt)
        T(ax, x + w / 2, 680, f"{wgt:.2f}", 30, ORANGE if k == 1 else MUTED, "bold", ha="center", alpha=a)
    T(ax, 960, 780, "attention weights of “it”: it mostly looks at “cat”", 30, BODY, ha="center", alpha=a)


def v_qkv(fig, ax, p):
    header(ax, "What attention computes", "Each token becomes three vectors", p)
    R(ax, 160, 470, 220, 120, "#FFFFFF", ec=LINE, lw=2, r=18)
    T(ax, 270, 530, "cat", 56, INK, "bold", ha="center", va="center")
    rows = [("Q", "Query", "what am I looking for?", BLUE),
            ("K", "Key", "what do I offer?", ORANGE),
            ("V", "Value", "the information I pass on", GREEN)]
    rng = np.random.default_rng(4)
    for i, (s, name, desc, c) in enumerate(rows):
        a = ease(p, 0.15 + i * 0.22, 0.3 + i * 0.22)
        y = 300 + i * 220
        arrow(ax, 390, 530, 560, y + 50, color=c, lw=4, alpha=a)
        T(ax, 590, y + 50, s, 60, c, "bold", va="center", alpha=a)
        vals = rng.uniform(0.25, 1, 12)
        for j, v in enumerate(vals):
            R(ax, 670 + j * 46, y + 20, 40, 60, c, alpha=a * v, r=6)
        T(ax, 1250, y + 22, name, 34, INK, "bold", alpha=a)
        T(ax, 1250, y + 66, desc, 28, BODY, alpha=a)
    T(ax, 670, 960, "each vector has D = 64 numbers (12 shown)", 24, MUTED, alpha=ease(p, .8, .95))


def v_three_steps(fig, ax, p):
    header(ax, "What attention computes", "Three steps: score → softmax → weighted average", p)
    steps = ["1  Score: Q · Kᵀ", "2  Softmax each row", "3  Weighted sum of V"]
    act = 0 if p < 0.36 else (1 if p < 0.62 else 2)
    for i, s in enumerate(steps):
        pill(ax, 128 + i * 470, 250, s, fc=[BLUE, ORANGE, GREEN][i] if i <= act else "#C9C7BF", size=26)
    n, cell, x0, y0 = 7, 74, 300, 400
    rng = np.random.default_rng(7)
    S = rng.normal(0, 1, (n, n))
    S[4, 1] = 3.2
    S[4, 4] = 0.5
    P = np.exp(S - S.max(1, keepdims=True))
    P /= P.sum(1, keepdims=True)
    for i, t in enumerate(TOKENS):
        T(ax, x0 - 16, y0 + i * cell + cell / 2, t, 22, ORANGE if t == "it" else BODY, "bold" if t == "it" else "normal", ha="right", va="center")
        T(ax, x0 + i * cell + cell / 2, y0 - 14, t, 22, BODY, ha="center", va="bottom", rotation=0)
    fill = ease(p, 0.02, 0.33) * n * n
    cmapS, cmapP = plt.get_cmap("Blues"), plt.get_cmap("Oranges")
    for i in range(n):
        for j in range(n):
            if i * n + j >= fill:
                R(ax, x0 + j * cell, y0 + i * cell, cell - 4, cell - 4, "#E9E7E0", r=6)
                continue
            if act >= 1:
                c = cmapP(0.15 + 0.85 * P[i, j] ** 0.6)
            else:
                c = cmapS(0.2 + 0.6 * (S[i, j] - S.min()) / (S.max() - S.min()))
            R(ax, x0 + j * cell, y0 + i * cell, cell - 4, cell - 4, c, r=6)
    T(ax, x0 + n * cell / 2, y0 + n * cell + 20, "N × N table of scores" if act == 0 else "rows of weights (each sums to 1)",
      24, MUTED, ha="center")
    if act >= 1:
        R(ax, x0 - 6, y0 + 4 * cell - 6, n * cell + 8, cell + 8, "none", ec=ORANGE, lw=4, r=8, z=6)
    a3 = ease(p, 0.64, 0.85)
    if a3 > 0:
        xr = 1020
        T(ax, xr, 420, "row of “it”  ×  value vectors  =  output", 28, INK, "bold", alpha=a3)
        for j in range(n):
            wgt = P[4, j]
            R(ax, xr, 480 + j * 56, 260 * wgt + 6, 40, ORANGE, alpha=a3, r=5)
            T(ax, xr + 280, 500 + j * 56, f"{TOKENS[j]}  {wgt:.2f}", 22, BODY, va="center", alpha=a3)
        arrow(ax, xr + 470, 680, xr + 560, 680, color=GREEN, lw=5, alpha=a3)
        for k in range(8):
            R(ax, xr + 590 + k * 30, 650, 26, 60, GREEN, alpha=a3 * (0.4 + 0.6 * ((k * 37) % 10) / 10), r=4)
        T(ax, xr + 590, 730, "new “it” ≈ mostly “cat”", 24, DGREEN, "bold", alpha=a3)


def v_formula(fig, ax, p):
    header(ax, "What attention computes", "The whole thing in one line", p)
    a = ease(p, 0.05, 0.25)
    T(ax, 960, 420, r"$O\;=\;\mathrm{softmax}\!\left(\frac{Q\,K^{\top}}{\sqrt{D}}\right)\,V$", 92, INK,
      ha="center", va="center", alpha=a)
    for i, (x, s, c) in enumerate([(700, "scores (N × N)", BLUE), (960, "weights", ORANGE), (1290, "weighted average", GREEN)]):
        b = ease(p, 0.25 + i * 0.1, 0.35 + i * 0.1)
        T(ax, x, 560, s, 28, c, "bold", ha="center", alpha=b)
    c = ease(p, 0.6, 0.8)
    R(ax, 360, 700, 1200, 170, NAVY, alpha=c, r=24)
    T(ax, 960, 750, "Attention kernel", 40, PAPER, "bold", ha="center", alpha=c)
    T(ax, 960, 810, "the low-level code that computes this as fast as possible on real hardware", 28, SOFT, ha="center", alpha=c)


# ================================================================== 3. cost
def v_matrix_grow(fig, ax, p):
    header(ax, "Why it gets expensive", "The score table grows with N²", p)
    Ns = [1024, 2048, 4096, 8192, 16384]
    mems = ["32 MB", "128 MB", "512 MB", "2 GB", "8 GB"]
    flops = ["2.1", "8.6", "34", "137", "550"]
    t = ease(p, 0.05, 0.8) * 4
    k = int(min(4, math.floor(t + 1e-9)))
    u = t - k
    side = 620 * (Ns[k] / 16384) * (1 + u) if k < 4 else 620
    side = min(side, 620)
    R(ax, 130, 300, 620, 620, "#ECEBE4", r=8)
    R(ax, 130, 300, side, side, BLUE, alpha=0.85, r=6)
    if side > 260:
        T(ax, 130 + side / 2, 300 + side / 2, f"N = {Ns[k]:,}", 40, PAPER, "bold", ha="center", va="center")
    else:
        T(ax, 130 + side + 14, 300 + side / 2, f"N = {Ns[k]:,}", 28, INK, "bold", va="center")
    T(ax, 130, 940, "every token × every token", 24, MUTED)
    T(ax, 930, 300, "N", 26, MUTED, "bold")
    T(ax, 1130, 300, "score memory (8 heads)", 26, MUTED, "bold")
    T(ax, 1520, 300, "GFLOP", 26, MUTED, "bold")
    for i in range(k + 1):
        y = 360 + i * 110
        a = 1.0 if i < k else ease(u, 0, 0.4) if k < 4 else 1
        T(ax, 930, y, f"{Ns[i]:,}", 40, INK, "bold", alpha=a)
        T(ax, 1130, y, mems[i], 40, ORANGE if i == 4 else INK, "bold", alpha=a)
        T(ax, 1520, y, flops[i], 40, INK, "bold", alpha=a)
        if i > 0:
            T(ax, 1400, y - 50, "×4", 24, ORANGE, "bold", alpha=a)


def v_cost_numbers(fig, ax, p):
    header(ax, "Why it gets expensive", "At 16,384 tokens and 8 heads", p)
    for i, (num, lab, c) in enumerate([("550", "billion floating-point operations\nper forward pass", BLUE),
                                       ("8 GB", "for the N × N score table\nalone (FP32)", ORANGE)]):
        a = ease(p, 0.05 + 0.2 * i, 0.2 + 0.2 * i)
        x = 128 + i * 840
        R(ax, x, 290, 800, 8, c, alpha=a)
        T(ax, x, 320, num, 150, INK, "bold", alpha=a)
        T(ax, x, 520, lab, 32, BODY, alpha=a, linespacing=1.4)
    a = ease(p, 0.55, 0.75)
    T(ax, 128, 720, "Memory of the test laptop", 28, INK, "bold", alpha=a)
    R(ax, 128, 770, 1664, 90, "#E4E2DA", alpha=a, r=12)
    R(ax, 128, 770, 1664 * 0.5 * ease(p, 0.65, 0.85), 90, ORANGE, alpha=a, r=12)
    T(ax, 128 + 24, 815, "8 GB score table", 28, PAPER, "bold", va="center", alpha=ease(p, .78, .9))
    T(ax, 1792 - 24, 815, "16 GB total", 28, BODY, "bold", ha="right", va="center", alpha=a)


# ================================================================== 4. block sparsity
def _band(nb, w=1):
    i = np.arange(nb)
    m = np.abs(i[:, None] - i[None, :]) <= w
    m[:, 0] = True
    m[0, :] = True
    return m


def v_tiles(fig, ax, p):
    header(ax, "Block sparsity", "Compute only the tiles that matter", p)
    nb, cell = 16, 40
    m = _band(nb)
    fade = ease(p, 0.35, 0.7)
    for i in range(nb):
        for j in range(nb):
            if m[i, j]:
                c = BLUE
                R(ax, 160 + j * cell, 280 + i * cell, cell - 4, cell - 4, c, r=5)
            else:
                R(ax, 160 + j * cell, 280 + i * cell, cell - 4, cell - 4, BLUE, alpha=1 - fade, r=5)
                R(ax, 160 + j * cell, 280 + i * cell, cell - 4, cell - 4, "#E4E2DA", alpha=fade, r=5)
    a = ease(p, 0.05, 0.2)
    T(ax, 900, 300, "each square = a 64 × 64 tile of scores", 30, INK, "bold", alpha=a)
    R(ax, 900, 400, 44, 44, BLUE, alpha=a, r=6)
    T(ax, 965, 422, "computed: a small dense matrix multiply", 28, BODY, va="center", alpha=a)
    b = ease(p, 0.45, 0.65)
    R(ax, 900, 480, 44, 44, "#E4E2DA", alpha=b, r=6)
    T(ax, 965, 502, "skipped: never touched — zero work", 28, BODY, va="center", alpha=b)
    kept = int(m.sum())
    T(ax, 900, 620, f"{kept if fade > 0.5 else 256} of 256 tiles computed", 44, ORANGE if fade > .5 else INK, "bold", alpha=a)
    T(ax, 900, 700, "a local window + a few global tiles", 26, MUTED, alpha=b)


def v_patterns(fig, ax, p):
    header(ax, "Block sparsity", "Five configurable patterns", p)
    names = ["dense", "local", "strided", "longformer", "random"]
    caps = ["everything", "neighbours\n(sliding window)", "neighbours +\nsummary columns", "neighbours +\nglobal blocks", "same number of\nrandom blocks per row"]
    nb, cell = 16, 19
    for i, (nm, cap) in enumerate(zip(names, caps)):
        a = ease(p, 0.04 + i * 0.15, 0.16 + i * 0.15)
        x = 150 + i * 340
        lay = PAT.make_pattern(nm, nb * 64, 64, target_density=0.25)
        for r in range(nb):
            for c in range(nb):
                R(ax, x + c * cell, 380 + r * cell, cell - 3, cell - 3, BLUE if lay[r, c] else "#E4E2DA", alpha=a, r=3)
        T(ax, x + nb * cell / 2, 330, nm, 32, INK, "bold", ha="center", alpha=a)
        T(ax, x + nb * cell / 2, 700, cap, 24, BODY, ha="center", alpha=a, linespacing=1.3)


def v_density(fig, ax, p):
    header(ax, "Block sparsity", "Density and the ideal speed-up", p)
    nb, cell = 8, 64
    rng = np.random.default_rng(2)
    m = np.zeros((nb, nb), bool)
    for r in range(nb):
        m[r, rng.integers(0, nb)] = True
    grid_tiles(ax, 160, 300, m, cell, gap=6)
    a, b, c = ease(p, .05, .2), ease(p, .3, .45), ease(p, .6, .78)
    T(ax, 820, 320, "density  =  kept / total  =  8 / 64  =  12.5%", 40, INK, "bold", alpha=a)
    T(ax, 820, 420, "ideal speed-up  =  1 / density  =  8×", 40, ORANGE, "bold", alpha=b)
    R(ax, 820, 560, 960, 200, NAVY, alpha=c, r=22)
    T(ax, 860, 600, "The central question", 30, "#F2A27E", "bold", alpha=c)
    T(ax, 860, 650, "How close does real hardware get to the ideal —\nand when it falls short, why?", 32, PAPER, alpha=c, linespacing=1.4)


def v_masked_vs_skip(fig, ax, p):
    header(ax, "Block sparsity", "Masking saves nothing — skipping does", p)
    nb, cell = 8, 58
    rng = np.random.default_rng(2)
    m = np.zeros((nb, nb), bool)
    for r in range(nb):
        m[r, rng.integers(0, nb)] = True
    t = ease(p, 0.08, 0.6)
    for side, (x0, title) in enumerate([(160, "Compute everything, then mask"), (1030, "Skip (our kernel)")]):
        T(ax, x0, 270, title, 32, INK, "bold")
        done = 0
        for i in range(nb):
            for j in range(nb):
                idx = i * nb + j
                visited = idx < t * nb * nb
                if side == 0:
                    if visited:
                        done += 1
                    col = (BLUE if m[i, j] else ORANGE) if visited else "#E4E2DA"
                    R(ax, x0 + j * cell, 340 + i * cell, cell - 5, cell - 5, col, alpha=0.9, r=5)
                    if visited and not m[i, j] and p > 0.62:
                        T(ax, x0 + j * cell + cell / 2 - 2, 340 + i * cell + cell / 2 - 2, "×", 30, PAPER, "bold", ha="center", va="center")
                else:
                    if visited and m[i, j]:
                        done += 1
                    R(ax, x0 + j * cell, 340 + i * cell, cell - 5, cell - 5, BLUE if (visited and m[i, j]) else "#E4E2DA", r=5)
        frac = done / (nb * nb)
        T(ax, x0, 830, "work done: " + f"{frac*100:.1f}".rstrip("0").rstrip(".") + "%", 40, ORANGE if side == 0 else BLUE, "bold")
    T(ax, 160, 900, "orange = computed, then thrown away", 24, MUTED, alpha=ease(p, .62, .75))



# ================================================================== 5. our kernel
KMASK = PAT.longformer(8, 1, 1)


def _grid(ax, x, y, mask, cell, row_hl=None, a=1.0):
    n = mask.shape[0]
    for i in range(n):
        for j in range(n):
            on = mask[i, j]
            c = (ORANGE if (row_hl == i and on) else BLUE) if on else "#E4E2DA"
            if row_hl is not None and i != row_hl and on:
                c = "#9DB5D9"
            R(ax, x + j * cell, y + i * cell, cell - 6, cell - 6, c, alpha=a, r=6)
    if row_hl is not None:
        R(ax, x - 6, y + row_hl * cell - 6, n * cell + 6, cell + 6, "none", ec=INK, lw=4, r=8, z=6)


def v_kernel_loop(fig, ax, p):
    header(ax, "Our kernel", "One query block at a time", p)
    n, cell = 8, 74
    row = min(n - 1, int(ease(p, 0.05, 0.8) * n))
    T(ax, 160, 255, "key blocks →", 24, MUTED)
    for i in range(n):
        T(ax, 145, 300 + i * cell + cell / 2 - 3, f"q{i}", 22, ORANGE if i == row else MUTED, "bold" if i == row else "normal",
          ha="right", va="center")
    _grid(ax, 160, 300, KMASK, cell, row_hl=row)
    cols = np.flatnonzero(KMASK[row])
    x0 = 900
    T(ax, x0, 300, f"query block {row} needs key blocks: {', '.join(map(str, cols))}", 30, INK, "bold")
    for k, c in enumerate(cols):
        R(ax, x0 + k * 90, 360, 78, 78, ORANGE, r=8)
        T(ax, x0 + k * 90 + 39, 399, f"K{c}", 24, PAPER, "bold", ha="center", va="center")
    steps = ["1  look up the kept key blocks (the plan)", "2  fetch only those keys and values",
             "3  scores → softmax → weighted sum", "4  write this block's output, next block"]
    for i, s in enumerate(steps):
        T(ax, x0, 500 + i * 70, s, 30, BODY if i else INK, "bold" if i == 2 else "normal")
    a = ease(p, 0.82, 0.95)
    R(ax, x0, 800, 880, 100, NAVY, alpha=a, r=18)
    T(ax, x0 + 440, 850, "grey blocks are never touched", 30, PAPER, "bold", ha="center", va="center", alpha=a)


def v_kernel_code(fig, ax, p):
    header(ax, "Our kernel", "Five lines of PyTorch", p)
    code = ["for each query block i (64 queries):",
            "    k_i, v_i = the kept key/value blocks      # from the plan",
            "    scores  = q_i @ k_iᵀ / sqrt(D)",
            "    weights = softmax(scores)",
            "    out_i   = weights @ v_i"]
    R(ax, 128, 270, 1664, 360, NAVY, r=22)
    hl = min(4, int(ease(p, 0.05, 0.45) * 5))
    for i, l in enumerate(code):
        if i == hl:
            R(ax, 140, 298 + i * 62, 1640, 56, NAVY2, r=8)
        T(ax, 170, 306 + i * 62, l, 32, "#E8ECF4", family=MONO)
    a, b = ease(p, 0.5, 0.62), ease(p, 0.66, 0.78)
    card(ax, 128, 700, 800, 200, "PyTorch does the arithmetic", "matrix multiply, softmax —\nlike any library", GREEN, alpha=a)
    card(ax, 992, 700, 800, 200, "Our code decides what is computed", "which blocks, how they are fetched —\nthat is where the savings come from", ORANGE, alpha=b)


def v_plan(fig, ax, p):
    header(ax, "Our kernel", "The plan: a list per query block", p)
    m = PAT.longformer(6, 1, 1)
    _grid(ax, 180, 300, m, 90)
    T(ax, 860, 255, "query block  →  key blocks it needs", 26, MUTED, alpha=ease(p, 0.02, 0.1))
    for i in range(6):
        a = ease(p, 0.08 + i * 0.1, 0.16 + i * 0.1)
        cols = np.flatnonzero(m[i])
        T(ax, 860, 310 + i * 90, f"block {i}  →  [{', '.join(map(str, cols))}]", 36, INK, alpha=a, family=MONO)
    b = ease(p, 0.75, 0.88)
    T(ax, 860, 880, "made once per pattern, reused for every input", 30, BLUE, "bold", alpha=b)


def v_exact(fig, ax, p):
    header(ax, "Our kernel", "Same answer as dense attention", p)
    a, b, c = ease(p, .05, .2), ease(p, .3, .45), ease(p, .55, .7)
    T(ax, 128, 290, "168 / 168", 170, INK, "bold", alpha=a)
    T(ax, 128, 510, "correctness checks pass", 40, BODY, alpha=a)
    for i, s in enumerate(["CPU", "GPU", "5 patterns", "4 shapes", "every version"]):
        pill(ax, 128 + i * 300, 620, s, fc=BLUE, size=28, alpha=b)
    T(ax, 128, 760, "compared with a float64 (high-precision) reference · largest error ≈ 1 in a million", 30, BODY, alpha=c)


# ================================================================== 6. optimizing
def v_copy_vs_view(fig, ax, p):
    header(ax, "Optimizing the kernel", "v1 copies the blocks; v2 points at them", p)
    need = [3, 4, 5]
    for panel, (y0, title, t0) in enumerate([(300, "v1  gather: copy the needed blocks", 0.0), (700, "v2  view: use them in place", 0.42)]):
        a = ease(p, t0, t0 + 0.08)
        T(ax, 128, y0 - 40, title, 32, INK, "bold", alpha=a)
        T(ax, 128, y0 + 20, "keys in\nmemory", 22, MUTED, alpha=a, linespacing=1.2)
        for k in range(10):
            c = ORANGE if k in need else "#C9D2E3"
            R(ax, 260 + k * 92, y0, 82, 82, c, alpha=a, r=8)
            T(ax, 301 + k * 92, y0 + 41, f"K{k}", 22, PAPER if k in need else INK, "bold", ha="center", va="center", alpha=a)
        if panel == 0:
            m = ease(p, 0.1, 0.3)
            for j, k in enumerate(need):
                x = 260 + k * 92 + (1300 - (260 + k * 92) + j * 92) * m
                y = y0 + 150 * m
                R(ax, x, y, 82, 82, ORANGE, alpha=a * 0.9, r=8)
            T(ax, 1300, y0 + 250, "new copy: extra time + memory", 26, OINK, "bold", alpha=ease(p, .28, .36))
            T(ax, 1300, y0 - 40, "CPU 32 ms · GPU 40 ms", 28, MUTED, "bold", alpha=ease(p, .3, .38))
        else:
            b = ease(p, 0.5, 0.62)
            R(ax, 260 + 3 * 92 - 8, y0 - 8, 3 * 92 + 6, 98, "none", ec=BLUE, lw=6, alpha=b, r=12, z=6)
            T(ax, 1300, y0 + 20, "no copy — just a pointer\nto K3..K5 (a 'view')", 28, BLUE, "bold", alpha=b, linespacing=1.3)
            T(ax, 1300, y0 - 40, "CPU 25 ms · GPU 15 ms", 28, DGREEN, "bold", alpha=ease(p, .62, .72))
    T(ax, 128, 900, "works because a sliding window's blocks sit next to each other in memory", 26, MUTED, alpha=ease(p, .75, .85))


def v_padded(fig, ax, p):
    header(ax, "Optimizing the kernel", "Tried and rejected: one big padded multiply", p)
    lens = [64, 9, 9, 9, 9, 9, 9, 8]
    g = ease(p, 0.3, 0.7)
    x0, y0, sx = 300, 300, 17
    for i, L in enumerate(lens):
        T(ax, x0 - 20, y0 + i * 66 + 25, "global row" if i == 0 else f"row {i}", 24, ORANGE if i == 0 else MUTED,
          "bold" if i == 0 else "normal", ha="right", va="center")
        R(ax, x0, y0 + i * 66, L * sx, 50, ORANGE if i == 0 else BLUE, r=6)
        if i > 0:
            R(ax, x0 + L * sx, y0 + i * 66, (64 - L) * sx * g, 50, "#E4E2DA", ec=RED, lw=1.5, r=6)
    a = ease(p, 0.05, 0.2)
    T(ax, x0 + 64 * sx + 40, 300, "Longformer: the global\nrow needs all 64 blocks,\nthe others about 9", 28, BODY, alpha=a, linespacing=1.35)
    b = ease(p, 0.55, 0.72)
    T(ax, x0 + 64 * sx + 40, 520, "padding every row\nto 64 = mostly\nwasted work", 32, RED, "bold", alpha=b, linespacing=1.3)
    c = ease(p, 0.75, 0.88)
    T(ax, x0 + 64 * sx + 40, 740, "317 ms vs 30 ms\n(10× slower)", 36, INK, "bold", alpha=c, linespacing=1.25)


def v_versions_chart(fig, ax, p):
    header(ax, "Optimizing the kernel", "Four versions, measured (N = 4,096, ≈12.5%)", p)
    rows = [("masked dense", 241.9, 65.2, PINK), ("v1 gather", 32.3, 40.4, VIOLET),
            ("v2 views (final)", 25.4, 15.1, BLUE), ("padded batch*", 317.4, 273.9, YELLOW)]
    g = ease(p, 0.05, 0.6)
    T(ax, 620, 270, "CPU", 28, INK, "bold")
    T(ax, 1240, 270, "GPU", 28, INK, "bold")
    for i, (lab, c_ms, g_ms, col) in enumerate(rows):
        y = 330 + i * 130
        T(ax, 560, y + 35, lab, 30, INK, "bold", ha="right", va="center")
        for x0, v in [(620, c_ms), (1240, g_ms)]:
            w = 520 * v / 400 * g
            R(ax, x0, y, max(w, 3), 70, col, r=8)
            T(ax, x0 + w + 12, y + 35, f"{v * g:.0f} ms", 28, INK, "bold", va="center")
    T(ax, 128, 880, "* padded batch measured on the Longformer pattern, the others on the local window", 24, MUTED,
      alpha=ease(p, .6, .7))


# ================================================================== 7. framework
def v_framework(fig, ax, p):
    header(ax, "The benchmarking framework", "One command runs everything", p)
    exps = ["correctness", "seqlen", "sparsity", "ablation", "blocksize", "threads", "memory"]
    a = ease(p, 0.03, 0.15)
    R(ax, 128, 290, 520, 110, NAVY, alpha=a, r=16)
    T(ax, 388, 345, "run_experiments.py --exp all", 26, "#E8ECF4", ha="center", va="center", family=MONO, alpha=a)
    for i, e in enumerate(exps):
        b = ease(p, 0.12 + i * 0.04, 0.2 + i * 0.04)
        R(ax, 720 + (i % 4) * 270, 290 + (i // 4) * 70, 250, 54, BLUE, alpha=b, r=27)
        T(ax, 845 + (i % 4) * 270, 317 + (i // 4) * 70, e, 24, PAPER, "bold", ha="center", va="center", alpha=b, family=MONO)
    c = ease(p, 0.45, 0.6)
    steps = [("results/*.csv\n+ logs", ORANGE), ("plot_results.py", GREEN), ("graphs", VIOLET), ("report + slides", BLUE)]
    for i, (s, col) in enumerate(steps):
        x = 128 + i * 430
        R(ax, x, 520, 360, 120, col, alpha=c, r=18)
        T(ax, x + 180, 580, s, 28, PAPER, "bold", ha="center", va="center", alpha=c, linespacing=1.2)
        if i < 3:
            arrow(ax, x + 365, 580, x + 425, 580, color=MUTED, lw=4, alpha=c)
    d = ease(p, 0.7, 0.85)
    T(ax, 128, 720, "configurable:", 28, INK, "bold", alpha=d)
    x = 128
    for s in ["pattern", "density", "sequence length", "block size", "heads", "device: cpu / mps"]:
        x += pill(ax, x, 770, s, fc="#FFFFFF", color=INK, size=26, alpha=d) + 18


def v_method(fig, ax, p):
    header(ax, "The benchmarking framework", "What we measure, and how", p)
    items = [("Time", "median of 7 runs\nafter a warm-up", BLUE), ("Throughput", "tokens per second\n= N ÷ time", GREEN),
             ("Computation", "FLOPs = 4·H·N²·D\nor only kept blocks", VIOLET), ("Memory", "one call, fresh process,\nOS peak memory", ORANGE)]
    for i, (t, b, c) in enumerate(items):
        a = ease(p, 0.05 + i * 0.16, 0.15 + i * 0.16)
        card(ax, 128 + i * 420, 300, 395, 230, t, b, c, alpha=a, tsize=36, bsize=26)
    d = ease(p, 0.75, 0.88)
    T(ax, 128, 640, "Apple M4 laptop: 4 fast + 6 efficiency CPU cores · 10-core GPU · 16 GB", 30, BODY, alpha=d)
    T(ax, 128, 700, "PyTorch 2.11 · 8 heads · 64 numbers per token · 64×64 blocks", 30, BODY, alpha=d)


# ================================================================== 8-9. sequence length & work
SEQ_ST = [("dense", "dense", "Dense", ORANGE, "s"), ("library (PyTorch SDPA)", "dense", "PyTorch built-in", GREEN, "^"),
          ("block-sparse", "local (±4 blocks)", "Ours: local window", BLUE, "o"),
          ("block-sparse", "longformer (±3 + 1 global)", "Ours: Longformer", VIOLET, "D")]


def _seq_ax(fig, fr_dense, fr_sparse, df=seq, ycol="time_ms", ylabel="milliseconds (log scale)", box=(210, 260, 1080, 660)):
    a2 = inset(fig, *box)
    for impl, pat, lab, c, mk in SEQ_ST:
        d = df[(df.impl == impl) & (df.pattern == pat)].sort_values("N")
        fr = fr_sparse if impl == "block-sparse" else fr_dense
        partial_line(a2, d.N, d[ycol], fr, color=c, lw=4, marker=mk, ms=11)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlim(400, 21000)
    logfmt(a2, "x")
    axlabels(a2, "sequence length N (tokens)", ylabel)
    hs = [matplotlib.lines.Line2D([], [], color=c, marker=mk, lw=4, ms=10, label=lab) for _, _, lab, c, mk in SEQ_ST]
    a2.legend(handles=hs, loc="upper left", fontsize=16, frameon=False)
    return a2


def v_seq_dense(fig, ax, p):
    header(ax, "Result 1 · sequence length (CPU)", "Dense grows quadratically", p)
    a2 = _seq_ax(fig, ease(p, 0.1, 0.75), 0.0)
    a2.set_ylim(0.5, 3000)
    a = ease(p, 0.5, 0.7)
    T(ax, 1360, 320, "log-log axes:\nslope = how fast\ncost grows", 30, BODY, alpha=a, linespacing=1.35)
    T(ax, 1360, 520, "dense: slope 2", 40, ORANGE, "bold", alpha=a)
    T(ax, 1360, 580, "double N → 4× time", 32, INK, alpha=a)


def v_seq_sparse(fig, ax, p):
    header(ax, "Result 1 · sequence length (CPU)", "Our window kernels grow linearly", p)
    a2 = _seq_ax(fig, 1.0, ease(p, 0.05, 0.6))
    a2.set_ylim(0.5, 3000)
    a = ease(p, 0.4, 0.6)
    T(ax, 1360, 320, "local & Longformer:\nslope 1", 40, BLUE, "bold", alpha=a, linespacing=1.15)
    T(ax, 1360, 470, "each query block\nalways sees the same\nnumber of blocks", 28, BODY, alpha=a, linespacing=1.35)


def v_seq_numbers(fig, ax, p):
    header(ax, "Result 1 · sequence length (CPU)", "The numbers", p)
    g = ease(p, 0.05, 0.45)
    T(ax, 128, 270, "8,192 tokens", 34, INK, "bold")
    for i, (lab, v, c) in enumerate([("Dense", 763.7, ORANGE), ("Ours (local window)", 52.0, BLUE)]):
        y = 330 + i * 90
        T(ax, 128, y + 30, lab, 28, BODY, va="center")
        R(ax, 480, y, max(1100 * v / 1400 * g, 3), 60, c, r=8)
        T(ax, 480 + 1100 * v / 1400 * g + 14, y + 30, f"{v * g:.0f} ms", 28, INK, "bold", va="center")
    a = ease(p, 0.38, 0.5)
    T(ax, 128, 520, "14.7× faster  ·  14.5× less work", 36, BLUE, "bold", alpha=a)
    b = ease(p, 0.55, 0.85)
    T(ax, 128, 620, "16,384 tokens", 34, INK, "bold", alpha=b)
    for i, (lab, v, c) in enumerate([("Dense", None, ORANGE), ("PyTorch built-in", 1377.7, GREEN), ("Ours (local window)", 104.2, BLUE)]):
        y = 680 + i * 80
        T(ax, 128, y + 26, lab, 28, BODY, va="center", alpha=b)
        if v is None:
            T(ax, 480, y + 26, "can't run — needs two 8 GB tables", 28, RED, "bold", va="center", alpha=b)
        else:
            w = 1100 * v / 1400 * ease(p, 0.6, 0.85)
            R(ax, 480, y, max(w, 3), 52, c, alpha=b, r=8)
            T(ax, 480 + w + 14, y + 26, f"{v:.0f} ms", 28, INK, "bold", va="center", alpha=b)


def v_work_chart(fig, ax, p):
    header(ax, "Result 2 · throughput and computation (CPU)", "Constant speed, far less work", p)
    fr = ease(p, 0.05, 0.55)
    a1 = _seq_ax(fig, fr, fr, ycol="tokens_per_s", ylabel="tokens per second (log)", box=(170, 270, 780, 560))
    a1.legend(loc="lower left", fontsize=14, frameon=False, handles=a1.get_legend().legend_handles)
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k"))
    a2 = inset(fig, 1080, 270, 700, 560)
    dd = seq[seq.impl == "library (PyTorch SDPA)"].sort_values("N")
    partial_line(a2, dd.N, dd.flops / 1e9, fr, color=ORANGE, lw=4, marker="s", ms=11)
    for pat, c, mk in [("local (±4 blocks)", BLUE, "o"), ("longformer (±3 + 1 global)", VIOLET, "D")]:
        d = seq[(seq.impl == "block-sparse") & (seq.pattern == pat)].sort_values("N")
        partial_line(a2, d.N, d.flops / 1e9, fr, color=c, lw=4, marker=mk, ms=11)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    logfmt(a2, "x")
    axlabels(a2, "sequence length N", "billions of operations (log)")
    T(ax, 170, 935, "throughput: ours ≈ 157k tokens/s at every length, dense falls to 10.7k", 24, BODY, alpha=ease(p, .55, .7))
    T(ax, 1080, 935, "16k tokens: 550 → 19 billion = 29× less", 24, BLUE, "bold", alpha=ease(p, .7, .85))


# ================================================================== 10. sparsity
def _spar_ax(fig, fr, fr_mask, df=spar, title_ideal=True):
    a2 = inset(fig, 210, 260, 1080, 660)
    d = df[(df.impl == "block-sparse") & (df.pattern == "random")].sort_values("density", ascending=False)
    partial_line(a2, d.density, d.ideal_speedup, fr, color=MUTED, lw=2.5, ls=":", marker="", ms=0)
    st = {"local": (BLUE, "o"), "strided": (YELLOW, "s"), "longformer": (VIOLET, "D"), "random": (DGREEN, "v")}
    for pat, (c, mk) in st.items():
        d = df[(df.impl == "block-sparse") & (df.pattern == pat)].sort_values("density", ascending=False)
        partial_line(a2, d.density, d.speedup_vs_dense, fr, color=c, lw=4, marker=mk, ms=11)
    if fr_mask > 0:
        d = df[df.impl == "masked dense"].sort_values("density", ascending=False)
        partial_line(a2, d.density, d.speedup_vs_dense, fr_mask, color=PINK, lw=4, ls="--", marker="v", ms=11)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log", base=2)
    a2.set_xlim(1.25, 0.025)
    a2.set_ylim(0.3, 40)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:g}%"))
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    axlabels(a2, "fraction of blocks computed  →  sparser", "speed-up over dense (log)")
    hs = [matplotlib.lines.Line2D([], [], color=MUTED, ls=":", lw=2.5, label="ideal = 1 / density")]
    hs += [matplotlib.lines.Line2D([], [], color=c, marker=mk, lw=4, ms=10, label=pat) for pat, (c, mk) in st.items()]
    if fr_mask > 0:
        hs.append(matplotlib.lines.Line2D([], [], color=PINK, ls="--", marker="v", lw=4, ms=10, label="masked dense"))
    a2.legend(handles=hs, loc="upper left", fontsize=16, frameon=False)
    return a2


def v_sparsity_speedup(fig, ax, p):
    header(ax, "Result 3 · sparsity level (CPU, N = 4,096)", "Speed-up follows the ideal", p)
    _spar_ax(fig, ease(p, 0.08, 0.65), 0)
    a, b = ease(p, 0.5, 0.65), ease(p, 0.7, 0.85)
    T(ax, 1360, 340, "12.5% of blocks", 30, INK, "bold", alpha=a)
    T(ax, 1360, 390, "5–6× faster", 40, BLUE, "bold", alpha=a)
    T(ax, 1360, 520, "≈3% of blocks", 30, INK, "bold", alpha=b)
    T(ax, 1360, 570, "≈12× faster", 40, BLUE, "bold", alpha=b)


def v_sparsity_masked(fig, ax, p):
    header(ax, "Result 3 · sparsity level (CPU, N = 4,096)", "Skipping helps — masking never does", p)
    _spar_ax(fig, 1.0, ease(p, 0.05, 0.5))
    a = ease(p, 0.45, 0.62)
    T(ax, 1360, 600, "masked dense:\n≈0.65× at\nevery density", 38, "#C2416E", "bold", alpha=a, linespacing=1.2)


# ================================================================== 11. memory
def v_memory_chart(fig, ax, p):
    header(ax, "Result 4 · memory (CPU)", "From gigabytes to megabytes", p)
    a2 = inset(fig, 210, 260, 1080, 660)
    fr = ease(p, 0.05, 0.5)
    st = [("dense", "Dense", ORANGE, "s"), ("masked_dense", "Masked dense", PINK, "v"),
          ("library", "PyTorch built-in", GREEN, "^"), ("v2_views", "Ours", BLUE, "o")]
    Ns = np.array([1024, 2048, 4096, 8192, 16384])
    a2.plot(Ns, 2 * 8 * Ns.astype(float) ** 2 * 4 / 2**20, color=MUTED, ls=":", lw=2.5)
    for impl, lab, c, mk in st:
        d = mem[mem.impl == impl].sort_values("N")
        partial_line(a2, d.N, d.extra_mb, fr, color=c, lw=4, marker=mk, ms=11)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_ylim(1, 30000)
    logfmt(a2, "x")
    axlabels(a2, "sequence length N (tokens)", "extra memory, MB (log)")
    hs = [matplotlib.lines.Line2D([], [], color=c, marker=mk, lw=4, ms=10, label=lab) for _, lab, c, mk in st]
    hs.append(matplotlib.lines.Line2D([], [], color=MUTED, ls=":", lw=2.5, label="prediction: two N×N tables"))
    a2.legend(handles=hs, loc="upper left", fontsize=16, frameon=False)
    a, b = ease(p, 0.35, 0.5), ease(p, 0.6, 0.75)
    T(ax, 1360, 320, "8,192 tokens", 30, INK, "bold", alpha=a)
    T(ax, 1360, 370, "dense 4,099 MB", 40, OINK, "bold", alpha=a)
    T(ax, 1360, 430, "prediction 4,096 MB", 26, BODY, alpha=a)
    T(ax, 1360, 520, "ours ≈ 5 MB", 40, BLUE, "bold", alpha=b)
    T(ax, 1360, 580, "at every length\n≈760× less", 28, BODY, alpha=b, linespacing=1.35)


# ================================================================== 12. block size & threads
def v_blocksize_chart(fig, ax, p):
    header(ax, "Result 5 · block size (±256-token window, N = 4,096)", "Precision vs efficiency", p)
    g = ease(p, 0.05, 0.5)
    for k, (col, lab, vals, fmt, c, y0, top) in enumerate([
            ("useful_fraction", "useful work", bsz.useful_fraction * 100, "{:.0f}%", BLUE, 300, 100),
            ("time_ms", "time", bsz.time_ms, "{:.0f} ms", VIOLET, 620, 55)]):
        a = ease(p, 0.05 + k * 0.3, 0.15 + k * 0.3)
        T(ax, 128, y0, lab, 32, INK, "bold", alpha=a)
        for i, (bs, v) in enumerate(zip(bsz.block_size, vals)):
            x = 420 + i * 360
            w = 200 * v / top * ease(p, 0.05 + k * 0.3, 0.3 + k * 0.3)
            R(ax, x, y0 + 50, w, 90, c, alpha=a, r=8)
            T(ax, x + w + 10, y0 + 95, fmt.format(v), 28, INK, "bold", va="center", alpha=a)
            T(ax, x, y0 + 165, f"block {bs}", 24, MUTED, alpha=a)
    b = ease(p, 0.75, 0.88)
    T(ax, 128, 900, "big blocks waste work but run faster (fewer, bigger multiplies) — we use 64", 28, INK, "bold", alpha=b)


def v_threads_chart(fig, ax, p):
    header(ax, "Result 5 · CPU threads (N = 4,096)", "Only the fast cores help", p)
    a2 = inset(fig, 210, 260, 1080, 660)
    a2.axvspan(4.5, 10.5, color=LINE, alpha=0.5)
    a2.plot([1, 10], [1, 10], color=MUTED, ls=":", lw=2)
    fr = ease(p, 0.05, 0.6)
    for impl, c, mk in [("dense", ORANGE, "s"), ("block-sparse", BLUE, "o")]:
        d = thr[thr.impl == impl].sort_values("threads")
        partial_line(a2, d.threads, d.speedup_vs_1_thread, fr, logx=False, logy=False, color=c, lw=4, marker=mk, ms=11)
    a2.set_xlim(0.5, 10.5)
    a2.set_ylim(0, 10.5)
    a2.set_xticks([1, 2, 4, 6, 8, 10])
    a2.text(7.5, 9.5, "efficiency cores", ha="center", fontsize=17, color=MUTED)
    axlabels(a2, "threads", "speed-up vs 1 thread")
    hs = [matplotlib.lines.Line2D([], [], color=ORANGE, marker="s", lw=4, ms=10, label="dense"),
          matplotlib.lines.Line2D([], [], color=BLUE, marker="o", lw=4, ms=10, label="ours"),
          matplotlib.lines.Line2D([], [], color=MUTED, ls=":", lw=2, label="perfect scaling")]
    a2.legend(handles=hs, loc="upper left", fontsize=16, frameon=False)
    a = ease(p, 0.5, 0.7)
    T(ax, 1360, 340, "4 threads =\n4 fast cores", 34, INK, "bold", alpha=a, linespacing=1.2)
    T(ax, 1360, 480, "2–2.4× faster", 40, BLUE, "bold", alpha=a)
    T(ax, 1360, 580, "6 slower cores\nadd nothing", 30, BODY, alpha=ease(p, .7, .85), linespacing=1.3)


# ================================================================== 13. GPU
def v_gpu_chart(fig, ax, p):
    header(ax, "Result 6 · the Apple GPU", "Same code, faster — and it still fits", p)
    a2 = inset(fig, 210, 260, 1080, 660)
    fr = ease(p, 0.05, 0.55)
    for df, ls, dev in [(seq, "-", "CPU"), (seq_g, "--", "GPU")]:
        for impl, pat, c, mk, lab in [("dense", "dense", ORANGE, "s", "dense"), ("block-sparse", "local (±4 blocks)", BLUE, "o", "ours")]:
            d = df[(df.impl == impl) & (df.pattern == pat)].sort_values("N")
            partial_line(a2, d.N, d.time_ms, fr, color=c, lw=4, ls=ls, marker=mk, ms=11)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlim(400, 21000)
    logfmt(a2, "x")
    axlabels(a2, "sequence length N (tokens)", "milliseconds (log)")
    hs = [matplotlib.lines.Line2D([], [], color=ORANGE, marker="s", lw=4, label="dense · CPU"),
          matplotlib.lines.Line2D([], [], color=BLUE, marker="o", lw=4, label="ours · CPU"),
          matplotlib.lines.Line2D([], [], color=ORANGE, marker="s", lw=4, ls="--", label="dense · GPU"),
          matplotlib.lines.Line2D([], [], color=BLUE, marker="o", lw=4, ls="--", label="ours · GPU")]
    a2.legend(handles=hs, loc="upper left", fontsize=16, frameon=False)
    a, b = ease(p, 0.3, 0.45), ease(p, 0.6, 0.75)
    T(ax, 1360, 320, "8,192 tokens (GPU)", 30, INK, "bold", alpha=a)
    T(ax, 1360, 370, "296 → 22 ms\n13× faster", 40, BLUE, "bold", alpha=a, linespacing=1.2)
    T(ax, 1360, 540, "16,384 tokens (GPU)", 30, INK, "bold", alpha=b)
    T(ax, 1360, 590, "PyTorch built-in:\nout of memory\nours: 34 ms", 34, RED, "bold", alpha=b, linespacing=1.25)


def v_gpu_floor(fig, ax, p):
    header(ax, "Result 6 · the Apple GPU (N = 4,096)", "A floor: every small GPU operation has a start-up cost", p)
    a2 = inset(fig, 210, 260, 1080, 660)
    fr = ease(p, 0.05, 0.6)
    for df, c, mk, lab in [(spar, BLUE, "o", "ours · CPU"), (spar_g, DGREEN, "D", "ours · GPU")]:
        d = df[(df.impl == "block-sparse") & (df.pattern == "local")].sort_values("density", ascending=False)
        partial_line(a2, d.density, d.time_ms, fr, color=c, lw=4, marker=mk, ms=11)
    a2.axhline(10, color=RED, ls="--", lw=2, alpha=ease(p, 0.55, 0.7))
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    a2.set_xlim(1.25, 0.035)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:g}%"))
    a2.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axlabels(a2, "fraction of blocks computed  →  sparser", "milliseconds (log)")
    hs = [matplotlib.lines.Line2D([], [], color=BLUE, marker="o", lw=4, label="ours · CPU"),
          matplotlib.lines.Line2D([], [], color=DGREEN, marker="D", lw=4, label="ours · GPU")]
    a2.legend(handles=hs, loc="upper right", fontsize=16, frameon=False)
    a = ease(p, 0.6, 0.75)
    T(ax, 1360, 340, "GPU stops getting\nfaster at ≈10 ms", 34, RED, "bold", alpha=a, linespacing=1.2)
    T(ax, 1360, 480, "a few small operations\nper query block, each\nwith a fixed launch cost", 28, BODY, alpha=a, linespacing=1.35)


# ================================================================== 14-15. why, conclusion
def v_why(fig, ax, p):
    header(ax, "Why not the ideal?", "Four computer-architecture reasons", p)
    items = [("Fixed cost per block", "a loop step and a small multiply\nthat don't shrink with the work", BLUE),
             ("Copying data costs time", "views instead of copies: 22% on\nCPU, 2.7× on GPU", ORANGE),
             ("Big beats small", "one large multiply keeps hardware\nbusier: sparse at 100% = 0.77× dense", GREEN),
             ("Uneven rows", "forcing them into one padded shape\nwastes work (10× on Longformer)", VIOLET)]
    for i, (t, b, c) in enumerate(items):
        a = ease(p, 0.04 + i * 0.2, 0.16 + i * 0.2)
        card(ax, 128 + (i % 2) * 840, 280 + (i // 2) * 320, 800, 280, t, b, c, alpha=a, tsize=34, bsize=28)


def v_conclusion(fig, ax, p):
    header(ax, "Conclusion", "Sparsity pays only when skipped blocks are never computed", p, dark=True)
    items = [("14.7×", "faster than dense\nat 8,192 tokens", BLUE), ("760×", "less memory\n(4,099 MB → 5 MB)", GREEN),
             ("0.65×", "masked dense: hiding\nthe skipped part saves nothing", ORANGE)]
    for i, (n, t, c) in enumerate(items):
        a = ease(p, 0.12 + i * 0.12, 0.24 + i * 0.12)
        x = 128 + i * 560
        R(ax, x, 300, 530, 8, c, alpha=a)
        T(ax, x, 330, n, 150, PAPER, "bold", alpha=a)
        T(ax, x, 540, t, 30, SOFT, alpha=a, linespacing=1.4)
    b = ease(p, 0.62, 0.8)
    R(ax, 128, 740, 1664, 160, NAVY2, alpha=b, r=20)
    T(ax, 170, 770, "The lesson:", 32, "#F2A27E", "bold", alpha=b)
    T(ax, 170, 825, "how you organise the remaining work decides how close you get to the ideal", 30, PAPER, alpha=b)


# ================================================================== registry
VIS = {
    "title": (v_title, True, True), "roadmap": (v_roadmap, True, False),
    "attn_sentence": (v_attn_sentence, True, False), "qkv": (v_qkv, True, False),
    "three_steps": (v_three_steps, True, False), "formula": (v_formula, True, False),
    "matrix_grow": (v_matrix_grow, True, False), "cost_numbers": (v_cost_numbers, True, False),
    "tiles": (v_tiles, True, False), "patterns": (v_patterns, True, False), "density": (v_density, True, False),
    "masked_vs_skip": (v_masked_vs_skip, True, False),
    "kernel_loop": (v_kernel_loop, True, False), "kernel_code": (v_kernel_code, True, False),
    "plan": (v_plan, True, False), "exact": (v_exact, True, False),
    "copy_vs_view": (v_copy_vs_view, True, False), "padded": (v_padded, True, False),
    "versions_chart": (v_versions_chart, True, False),
    "framework": (v_framework, True, False), "method": (v_method, True, False),
    "seq_dense": (v_seq_dense, True, False), "seq_sparse": (v_seq_sparse, True, False),
    "seq_numbers": (v_seq_numbers, True, False), "work_chart": (v_work_chart, True, False),
    "sparsity_speedup": (v_sparsity_speedup, True, False), "sparsity_masked": (v_sparsity_masked, True, False),
    "memory_chart": (v_memory_chart, True, False),
    "blocksize_chart": (v_blocksize_chart, True, False), "threads_chart": (v_threads_chart, True, False),
    "gpu_chart": (v_gpu_chart, True, False), "gpu_floor": (v_gpu_floor, True, False),
    "why": (v_why, True, False), "conclusion": (v_conclusion, True, True), "end": (v_end, True, True),
}


def render(fig, key, p, info):
    fn, _, dark = VIS[key]
    fig.clf()
    fig.patch.set_facecolor(NAVY if dark else PAPER)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_facecolor(NAVY if dark else PAPER)
    fn(fig, ax, p)
    if info is not None:
        chrome(ax, info, dark)
