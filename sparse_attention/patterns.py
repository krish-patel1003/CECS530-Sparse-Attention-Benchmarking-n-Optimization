"""Sparsity patterns.

We cut the N x N attention matrix into square blocks of BS x BS tokens.
A *pattern* is a small True/False grid of shape (num_blocks, num_blocks):
    mask[i, j] = True  ->  query block i looks at key block j  (we compute it)
    mask[i, j] = False ->  skipped (we never compute it)

Patterns supported
------------------
dense       every block                                   (the baseline)
local       a sliding window: each block sees its neighbours        (Longformer's local part)
strided     the window + every s-th block ("summary" blocks)        (Sparse Transformer)
longformer  the window + a few global blocks that see / are seen by all  (Longformer / BigBird)
random      the same number of random blocks in every row           (a simple random pattern)
"""
from __future__ import annotations

import numpy as np

PATTERNS = ["dense", "local", "strided", "longformer", "random"]


def num_blocks(N: int, BS: int) -> int:
    assert N % BS == 0, "sequence length must be a multiple of the block size"
    return N // BS


# ---------------------------------------------------------------- the patterns
def dense(nb):
    return np.ones((nb, nb), dtype=bool)


def local(nb, window):
    """Each query block sees key blocks within `window` blocks of itself."""
    i = np.arange(nb)
    return np.abs(i[:, None] - i[None, :]) <= window


def strided(nb, window, stride):
    """Local window + every `stride`-th key block."""
    mask = local(nb, window)
    mask[:, stride - 1::stride] = True
    return mask


def longformer(nb, window, n_global):
    """Local window + the first `n_global` blocks are global (see all, seen by all)."""
    mask = local(nb, window)
    mask[:n_global, :] = True
    mask[:, :n_global] = True
    return mask


def random_blocks(nb, per_row, seed=0):
    """Exactly `per_row` blocks in each row: the diagonal block plus random others."""
    rng = np.random.default_rng(seed)
    mask = np.eye(nb, dtype=bool)
    for row in range(nb):
        others = np.delete(np.arange(nb), row)
        mask[row, rng.choice(others, size=per_row - 1, replace=False)] = True
    return mask


# ---------------------------------------------------------------- helpers
def density(mask) -> float:
    """Fraction of blocks that are computed (1.0 = dense)."""
    return float(mask.mean())


def make_pattern(name: str, N: int, BS: int, target_density: float | None = None, **params):
    """Build a pattern by name.  Give either explicit parameters or a target density,
    in which case the pattern's size parameter is chosen to get as close as possible."""
    nb = num_blocks(N, BS)
    if name == "dense":
        return dense(nb)
    if target_density is not None:
        params = _params_for_density(name, nb, target_density)
    if name == "local":
        return local(nb, params.get("window", 2))
    if name == "strided":
        return strided(nb, params.get("window", 1), params.get("stride", 8))
    if name == "longformer":
        return longformer(nb, params.get("window", 2), params.get("n_global", 1))
    if name == "random":
        return random_blocks(nb, params.get("per_row", 4))
    raise ValueError(f"unknown pattern {name!r}, choose from {PATTERNS}")


def _params_for_density(name, nb, target):
    """Try every value of the pattern's size parameter and keep the closest density."""
    if name == "random":
        return {"per_row": int(np.clip(round(target * nb), 1, nb))}
    candidates = []
    if name == "local":
        candidates = [{"window": w} for w in range(nb)]
    elif name == "strided":
        candidates = [{"window": w, "stride": s} for w in range(0, 4) for s in range(2, nb + 1)]
    elif name == "longformer":
        candidates = [{"window": w, "n_global": 1} for w in range(nb)]
    build = {"local": lambda p: local(nb, **p), "strided": lambda p: strided(nb, **p),
             "longformer": lambda p: longformer(nb, **p)}[name]
    return min(candidates, key=lambda p: abs(density(build(p)) - target))


def blocks_per_row(mask):
    """For each query block: the list of key blocks it needs (the 'block list')."""
    return [np.flatnonzero(row) for row in mask]


def token_mask(mask, BS):
    """Expand a block pattern to a full N x N True/False mask (used by masked dense)."""
    return np.kron(mask, np.ones((BS, BS), dtype=bool))
