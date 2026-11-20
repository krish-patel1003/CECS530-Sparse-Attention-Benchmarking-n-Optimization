"""Measure how much EXTRA memory one attention call needs, in a brand-new process.

The operating system remembers the highest memory a process ever used ("peak resident set
size").  We read it just before and just after one call; the difference is what the call needed.

    python memory_probe.py <impl> <N> <heads> <D> <BS> <pattern>     -> prints one JSON line
"""
import json
import resource
import sys

import torch

from sparse_attention import attention as A, patterns as P


def peak_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss      # bytes on macOS


def main():
    impl, N, H, D, BS, pattern = sys.argv[1], *map(int, sys.argv[2:6]), sys.argv[6]
    torch.manual_seed(0)
    q, k, v = (torch.randn(H, N, D) for _ in range(3))
    mask = P.make_pattern(pattern, N, BS, window=4) if pattern == "local" else P.make_pattern(pattern, N, BS)
    plan = A.SparsePlan(mask, BS)
    tmask = torch.as_tensor(P.token_mask(mask, BS)) if impl == "masked_dense" else None
    before = peak_bytes()
    if impl == "dense":
        out = A.dense_attention(q, k, v)
    elif impl == "masked_dense":
        out = A.masked_dense_attention(q, k, v, tmask)
    elif impl == "library":
        out = A.library_attention(q, k, v)
    else:
        out = A.SPARSE_VERSIONS[impl](q, k, v, plan)
    extra = max(0, peak_bytes() - before - out.numel() * 4)       # don't count the output itself
    print(json.dumps(dict(impl=impl, N=N, extra_mb=extra / 2**20, density=P.density(mask))))


if __name__ == "__main__":
    main()
