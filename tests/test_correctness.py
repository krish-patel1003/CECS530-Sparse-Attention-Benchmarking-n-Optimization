"""Every attention version must give the same answer as a high-precision reference.

Run:  python tests/test_correctness.py      (also writes results/correctness.csv)
  or: python -m pytest -q tests
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sparse_attention import attention as A, patterns as P  # noqa: E402

TOL = 1e-4
CASES = [(4, 512, 64, 64), (2, 1024, 32, 32), (2, 2048, 64, 128), (8, 1024, 64, 16)]  # (H, N, D, BS)


def reference(q, k, v, mask, BS):
    """float64 masked dense attention = the ground truth."""
    tm = torch.as_tensor(P.token_mask(mask, BS))
    return A.masked_dense_attention(q.double(), k.double(), v.double(), tm).float()


def run_all(device="cpu"):
    rows = []
    for H, N, D, BS in CASES:
        g = torch.Generator().manual_seed(0)
        q, k, v = (torch.randn(H, N, D, generator=g) for _ in range(3))
        for name in P.PATTERNS:
            mask = P.make_pattern(name, N, BS, target_density=0.3)
            ref = reference(q, k, v, mask, BS)
            plan = A.SparsePlan(mask, BS, device)
            qd, kd, vd = (x.to(device) for x in (q, k, v))
            impls = dict(A.SPARSE_VERSIONS)
            impls["masked_dense"] = lambda a, b, c, _p: A.masked_dense_attention(
                a, b, c, torch.as_tensor(P.token_mask(mask, BS), device=device))
            if name == "dense":
                impls["dense"] = lambda a, b, c, _p: A.dense_attention(a, b, c)
            for impl, fn in impls.items():
                err = float((fn(qd, kd, vd, plan).cpu() - ref).abs().max())
                rows.append(dict(device=device, impl=impl, H=H, N=N, D=D, BS=BS, pattern=name,
                                 density=round(P.density(mask), 3), max_abs_error=err, passed=err < TOL))
    return rows


def test_all_versions_match_reference():
    bad = [r for r in run_all("cpu") if not r["passed"]]
    assert not bad, bad[:3]


def test_patterns_hit_target_density():
    for name in ["local", "strided", "longformer", "random"]:
        d = P.density(P.make_pattern(name, 4096, 64, target_density=0.25))
        assert abs(d - 0.25) < 0.06, (name, d)


if __name__ == "__main__":
    import pandas as pd
    rows = run_all("cpu")
    if torch.backends.mps.is_available():
        rows += run_all("mps")
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "correctness.csv", index=False)
    print(df.groupby(["device", "impl"]).agg(checks=("passed", "size"), passed=("passed", "sum"),
                                              worst_error=("max_abs_error", "max")).to_string())
    print(f"\nALL PASSED: {bool(df.passed.all())}  ({len(df)} checks)")
