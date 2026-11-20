"""Benchmarking framework: dense vs block-sparse attention.

    python run_experiments.py --list
    python run_experiments.py --exp all                  # everything (~10 minutes)
    python run_experiments.py --exp seqlen sparsity      # some experiments
    python run_experiments.py --exp seqlen --device mps  # the same experiment on the Apple GPU
    python run_experiments.py --exp sparsity --heads 4 --bs 32   # change the configuration

Each experiment writes results/<name>.csv and a console log results/logs/<name>.txt.
Measured: execution time, throughput (tokens/s), GFLOP/s, FLOPs (computation), memory,
and from those the speed-up and the computational reduction.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sparse_attention import attention as A, patterns as P

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
pd.set_option("display.width", 220)
ARGS = None


# ============================================================ measuring helpers
def sync():
    if ARGS.device == "mps":
        torch.mps.synchronize()          # wait for the GPU to finish before stopping the clock


def measure(fn, reps=7):
    """Run once to warm up, then `reps` times; return the MEDIAN time in milliseconds."""
    fn()
    sync()
    first = time.perf_counter()
    fn()
    sync()
    if time.perf_counter() - first > 1.5:   # very slow case: fewer repetitions
        reps = 3
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def inputs(N):
    g = torch.Generator().manual_seed(0)
    return tuple(torch.randn(ARGS.heads, N, ARGS.dim, generator=g).to(ARGS.device) for _ in range(3))


def record(impl, N, ms, flops, mask=None, **extra):
    """One result row with all the derived metrics."""
    dense = A.dense_flops(ARGS.heads, N, ARGS.dim)
    return dict(impl=impl, device=ARGS.device, N=N, heads=ARGS.heads, D=ARGS.dim, BS=ARGS.bs,
                density=1.0 if mask is None else P.density(mask), time_ms=ms,
                tokens_per_s=N / (ms / 1000), gflops=flops / (ms / 1000) / 1e9,
                flops=flops, flop_reduction=dense / flops, **extra)


@contextlib.contextmanager
def logged(name):
    """Print to the screen AND save a copy in results/logs/<name>.txt."""
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    buf, old = io.StringIO(), sys.stdout

    class Both(io.TextIOBase):
        def write(self, s):
            old.write(s)
            buf.write(s)
            return len(s)

    sys.stdout = Both()
    try:
        yield
    finally:
        sys.stdout = old
        (RESULTS / "logs" / f"{name}.txt").write_text(buf.getvalue())


def save(name, rows, cols):
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"{name}.csv", index=False)
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    return df


def banner(text):
    print("=" * 96)
    print(f"{text}\n  device={ARGS.device}  heads={ARGS.heads}  D={ARGS.dim}  block={ARGS.bs}  "
          f"torch threads={torch.get_num_threads()}")
    print("=" * 96)


def suffix():
    return "" if ARGS.device == "cpu" else f"_{ARGS.device}"


# ============================================================ experiments
def exp_correctness():
    """All versions vs a float64 reference."""
    banner("Correctness: every version vs a float64 reference (tolerance 1e-4)")
    sys.path.insert(0, str(ROOT / "tests"))
    import test_correctness as tc
    rows = tc.run_all("cpu") + (tc.run_all("mps") if torch.backends.mps.is_available() else [])
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "correctness.csv", index=False)
    print(df.groupby(["device", "impl"]).agg(checks=("passed", "size"), passed=("passed", "sum"),
                                              worst_error=("max_abs_error", "max")).to_string())
    print(f"\nALL PASSED: {bool(df.passed.all())}  ({len(df)} checks)")


def _sparse_cases(N):
    nb = N // ARGS.bs
    return {"local (±4 blocks)": P.local(nb, 4),                  # fixed window -> linear in N
            "longformer (±3 + 1 global)": P.longformer(nb, 3, 1),
            "random 12.5%": P.random_blocks(nb, max(1, nb // 8))}  # fixed fraction -> still N²


def exp_seqlen():
    """Time, throughput and FLOPs as the sequence gets longer."""
    banner("Sequence length: dense vs block-sparse as N grows")
    rows = []
    for N in [512, 1024, 2048, 4096, 8192, 16384]:
        q, k, v = inputs(N)
        H, D = ARGS.heads, ARGS.dim
        if N <= 8192:                       # the N x N matrix at 16k would need 8 GB (x2)
            rows.append(record("dense", N, measure(lambda: A.dense_attention(q, k, v)), A.dense_flops(H, N, D),
                               pattern="dense"))
        try:
            rows.append(record("library (PyTorch SDPA)", N, measure(lambda: A.library_attention(q, k, v)),
                               A.dense_flops(H, N, D), pattern="dense"))
        except RuntimeError as e:                       # e.g. the GPU cannot fit an N x N buffer
            if "out of memory" not in str(e):
                raise
            print(f"  OUT OF MEMORY: PyTorch SDPA at N={N} ({str(e).split('Tried to allocate')[-1].split(' on')[0].strip()} requested)")
            if ARGS.device == "mps":
                torch.mps.empty_cache()
        for name, mask in _sparse_cases(N).items():
            plan = A.SparsePlan(mask, ARGS.bs, ARGS.device)
            ms = measure(lambda: A.block_sparse_attention(q, k, v, plan))
            rows.append(record("block-sparse", N, ms, A.sparse_flops(H, plan.computed_blocks, ARGS.bs, D), mask, pattern=name))
        print(f"  N={N} done", flush=True)
    df = pd.DataFrame(rows)
    dense_t = df[df.impl == "dense"].set_index("N").time_ms
    df["speedup_vs_dense"] = [dense_t.get(n, np.nan) / t for n, t in zip(df.N, df.time_ms)]
    save("seqlen" + suffix(), df.to_dict("records"),
         ["impl", "pattern", "N", "density", "time_ms", "tokens_per_s", "flop_reduction", "speedup_vs_dense"])


def exp_sparsity():
    """Speed-up as the sparsity level changes, for every pattern."""
    N = 4096
    banner(f"Sparsity level: N={N}, block density 100% -> 3%")
    q, k, v = inputs(N)
    H, D = ARGS.heads, ARGS.dim
    dense_ms = measure(lambda: A.dense_attention(q, k, v))
    rows = [record("dense", N, dense_ms, A.dense_flops(H, N, D), pattern="dense", target=1.0)]
    for target in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
        for name in ["local", "strided", "longformer", "random"]:
            mask = P.make_pattern(name, N, ARGS.bs, target_density=target)
            plan = A.SparsePlan(mask, ARGS.bs, ARGS.device)
            ms = measure(lambda: A.block_sparse_attention(q, k, v, plan))
            rows.append(record("block-sparse", N, ms, A.sparse_flops(H, plan.computed_blocks, ARGS.bs, D), mask,
                               pattern=name, target=target))
        mask = P.make_pattern("random", N, ARGS.bs, target_density=target)
        tm = torch.as_tensor(P.token_mask(mask, ARGS.bs), device=ARGS.device)
        rows.append(record("masked dense", N, measure(lambda: A.masked_dense_attention(q, k, v, tm)),
                           A.dense_flops(H, N, D), mask, pattern="random", target=target))
        print(f"  density {target} done", flush=True)
    for r in rows:
        r["speedup_vs_dense"] = dense_ms / r["time_ms"]
        r["ideal_speedup"] = 1 / r["density"]
        r["percent_of_ideal"] = 100 * r["speedup_vs_dense"] / r["ideal_speedup"]
    save("sparsity" + suffix(), rows, ["impl", "pattern", "density", "time_ms", "speedup_vs_dense", "ideal_speedup",
                                       "percent_of_ideal", "flop_reduction"])


def exp_ablation():
    """Which implementation choices matter (masked dense -> v1 -> v2, plus the padded try)."""
    banner("Ablation: masked dense, v1 gather, v2 views (final), padded batched (rejected)")
    rows = []
    H, D = ARGS.heads, ARGS.dim
    for N in [4096, 16384]:
        q, k, v = inputs(N)
        nb = N // ARGS.bs
        cases = {"local": P.local(nb, 4), "longformer": P.longformer(nb, 4, 1), "random": P.random_blocks(nb, 9)}
        if N == 4096:
            cases = {n: P.make_pattern(n, N, ARGS.bs, target_density=0.125) for n in cases}
        for name, mask in cases.items():
            plan = A.SparsePlan(mask, ARGS.bs, ARGS.device)
            fl = A.sparse_flops(H, plan.computed_blocks, ARGS.bs, D)
            if N <= 8192:
                tm = torch.as_tensor(P.token_mask(mask, ARGS.bs), device=ARGS.device)
                rows.append(record("masked dense", N, measure(lambda: A.masked_dense_attention(q, k, v, tm)),
                                   A.dense_flops(H, N, D), mask, pattern=name))
            for impl, fn in A.SPARSE_VERSIONS.items():
                padded_bytes = 4 * H * plan.nb * plan.R * ARGS.bs * (ARGS.bs + 2 * D)
                if impl == "padded_batched" and padded_bytes > 3 * 2**30:
                    print(f"  skip padded_batched {name} N={N}: padding would need {padded_bytes / 2**30:.1f} GB "
                          f"(longest row {plan.R} blocks vs average {plan.counts.mean():.1f})")
                    continue
                ms = measure(lambda: fn(q, k, v, plan))
                padded_fl = A.sparse_flops(H, plan.nb * plan.R, ARGS.bs, D)
                rows.append(record(impl, N, ms, padded_fl if impl == "padded_batched" else fl, mask, pattern=name,
                                   longest_row=plan.R, average_row=float(plan.counts.mean())))
        print(f"  N={N} done", flush=True)
    save("ablation" + suffix(), rows, ["impl", "pattern", "N", "density", "time_ms", "gflops", "flops"])


def exp_blocksize():
    """Small vs large blocks for a 'token-level' pattern (a ±256-token sliding window)."""
    N = 4096
    banner(f"Block size: N={N}, sliding window of ±256 tokens")
    q, k, v = inputs(N)
    idx = np.arange(N)
    want = np.abs(idx[:, None] - idx[None, :]) <= 256          # the entries we actually want
    rows = []
    for BS in [16, 32, 64, 128]:
        nb = N // BS
        mask = want.reshape(nb, BS, nb, BS).any(axis=(1, 3))     # every block touching the band
        plan = A.SparsePlan(mask, BS, ARGS.device)
        ms = measure(lambda: A.block_sparse_attention(q, k, v, plan))
        fl = A.sparse_flops(ARGS.heads, plan.computed_blocks, BS, ARGS.dim)
        r = record("block-sparse", N, ms, fl, mask, block_size=BS, num_blocks=nb * nb,
                   useful_fraction=want.sum() / (plan.computed_blocks * BS * BS))
        r["BS"] = BS
        rows.append(r)
    save("blocksize" + suffix(), rows, ["block_size", "density", "useful_fraction", "time_ms", "gflops"])


def exp_threads():
    """CPU threads: how much do more cores help?"""
    N = 4096
    banner(f"CPU threads: N={N} (Apple M4 = 4 performance + 6 efficiency cores)")
    q, k, v = inputs(N)
    mask = P.make_pattern("local", N, ARGS.bs, target_density=0.125)
    plan = A.SparsePlan(mask, ARGS.bs)
    default = torch.get_num_threads()
    rows = []
    for t in [1, 2, 4, 6, 8, 10]:
        torch.set_num_threads(t)
        rows.append(record("dense", N, measure(lambda: A.dense_attention(q, k, v)), A.dense_flops(ARGS.heads, N, ARGS.dim),
                           threads=t, pattern="dense"))
        rows.append(record("block-sparse", N, measure(lambda: A.block_sparse_attention(q, k, v, plan)),
                           A.sparse_flops(ARGS.heads, plan.computed_blocks, ARGS.bs, ARGS.dim), mask, threads=t, pattern="local"))
    torch.set_num_threads(default)
    df = pd.DataFrame(rows)
    one = df[df.threads == 1].set_index("impl").time_ms
    df["speedup_vs_1_thread"] = [one[i] / t for i, t in zip(df.impl, df.time_ms)]
    save("threads", df.to_dict("records"), ["impl", "threads", "time_ms", "gflops", "speedup_vs_1_thread"])


def exp_memory():
    """Peak extra memory of one call, measured in a fresh process each time."""
    banner("Memory: peak extra memory of ONE call (fresh process per measurement)")
    rows = []
    for N in [1024, 2048, 4096, 8192, 16384]:
        for impl in ["dense", "masked_dense", "library", "v2_views"]:
            if impl in ("dense", "masked_dense") and N > 8192:
                continue
            out = subprocess.run([sys.executable, str(ROOT / "memory_probe.py"), impl, str(N), str(ARGS.heads),
                                  str(ARGS.dim), str(ARGS.bs), "local"], capture_output=True, text=True, cwd=ROOT)
            r = json.loads(out.stdout.strip().splitlines()[-1])
            r["predicted_scores_mb"] = A.dense_score_bytes(ARGS.heads, N) / 2**20
            r["inputs_mb"] = 3 * ARGS.heads * N * ARGS.dim * 4 / 2**20
            rows.append(r)
        print(f"  N={N} done", flush=True)
    save("memory", rows, ["impl", "N", "density", "extra_mb", "predicted_scores_mb", "inputs_mb"])


EXPERIMENTS = {"correctness": exp_correctness, "seqlen": exp_seqlen, "sparsity": exp_sparsity,
               "ablation": exp_ablation, "blocksize": exp_blocksize, "threads": exp_threads, "memory": exp_memory}


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+", default=["all"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--list", action="store_true")
    ARGS = ap.parse_args()
    if ARGS.list:
        for name, fn in EXPERIMENTS.items():
            print(f"{name:12s} {fn.__doc__.strip()}")
        return
    RESULTS.mkdir(exist_ok=True)
    names = list(EXPERIMENTS) if "all" in ARGS.exp else ARGS.exp
    for name in names:
        with logged(name + suffix()):
            EXPERIMENTS[name]()
        print()


if __name__ == "__main__":
    main()
