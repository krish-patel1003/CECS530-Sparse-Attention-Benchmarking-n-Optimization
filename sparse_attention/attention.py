"""Dense and block-sparse attention, written with PyTorch tensor operations.

All functions take q, k, v of shape [H, N, D]:  H heads, N tokens, D numbers per token.
They return the attention output, also [H, N, D].

    attention(Q, K, V) = softmax(Q Kᵀ / √D) · V

Implementations
---------------
dense_attention           the standard version: builds the full N x N score matrix
masked_dense_attention    dense + a sparsity mask applied AFTER computing everything
sparse_v1_gather          block-sparse v1: per query block, COPY the kept key/value blocks
                          together, then run ordinary attention on that small piece
sparse_v2_views           block-sparse v2 (our final kernel): same, but blocks that sit next to
                          each other are used in place ("views"), so nothing is copied
sparse_padded_batched     an alternative we tried: one big batched multiply, padding every row
                          to the longest row (slow when rows are uneven - kept for the ablation)

The block-sparse idea in one sentence:
    for each block of 64 queries, gather ONLY the key/value blocks the pattern keeps,
    and run ordinary attention on that small piece.
Skipped blocks are never touched, so the work (and memory) shrinks with the density.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import patterns as P


# ============================================================ dense baselines
def dense_attention(q, k, v):
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale        # [H, N, N]  <- the big matrix
    weights = torch.softmax(scores, dim=-1)           # each row sums to 1
    return weights @ v                                # [H, N, D]


def masked_dense_attention(q, k, v, token_mask):
    """Compute the FULL score matrix, then hide the entries the pattern skips.
    Correct output, but no work is saved: this is the 'naive sparse' baseline."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(-1, -2)) * scale
    scores = scores.masked_fill(~token_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def library_attention(q, k, v):
    """PyTorch's built-in fused attention (reference point, not our code)."""
    return torch.nn.functional.scaled_dot_product_attention(q[None], k[None], v[None])[0]


# ============================================================ the sparse plan
def _runs(cols):
    runs, start, prev = [], cols[0], cols[0]
    for c in cols[1:]:
        if c == prev + 1:
            prev = c
            continue
        runs.append((int(start), int(prev) + 1))
        start = prev = c
    runs.append((int(start), int(prev) + 1))
    return runs


class SparsePlan:
    """Pre-computed index tensors for one pattern (built once, reused for every call,
    just like a model reuses its attention pattern for every input)."""

    def __init__(self, mask: np.ndarray, BS: int, device="cpu"):
        self.mask, self.BS = mask, BS
        self.nb = mask.shape[0]
        rows = P.blocks_per_row(mask)                     # key blocks needed by each query block
        self.rows = [torch.as_tensor(r, device=device) for r in rows]
        self.counts = np.array([len(r) for r in rows])

        # --- padded batched version: pad every row to the longest row --------------
        R = int(self.counts.max())
        pad_idx = np.zeros((self.nb, R), dtype=np.int64)
        valid = np.zeros((self.nb, R), dtype=bool)
        for i, r in enumerate(rows):
            pad_idx[i, :len(r)] = r
            valid[i, :len(r)] = True
        self.R = R
        self.pad_idx = torch.as_tensor(pad_idx, device=device)
        # a key position is valid if its block is real (not padding): shape [nb, 1, R*BS]
        self.pad_valid = torch.as_tensor(np.repeat(valid, BS, axis=1)[:, None, :], device=device)

        # --- v2 (views): split each row's block list into runs of consecutive blocks ---
        #     e.g. blocks [3,4,5,9] -> runs [(3,6), (9,10)]; a run is used in place, no copy
        self.runs = [_runs(r) for r in rows]

    @property
    def computed_blocks(self):
        return int(self.counts.sum())


# ============================================================ block-sparse versions
def sparse_v1_gather(q, k, v, plan: SparsePlan):
    """v1: one small attention per query block.  The kept key/value blocks are gathered
    (copied) into one tensor, then: scores -> softmax -> weighted sum, as usual."""
    H, N, D = q.shape
    BS, scale = plan.BS, 1.0 / math.sqrt(D)
    kb = k.view(H, plan.nb, BS, D)                     # keys split into blocks
    vb = v.view(H, plan.nb, BS, D)
    out = torch.empty_like(q)
    for i, cols in enumerate(plan.rows):
        q_i = q[:, i * BS:(i + 1) * BS]                # this query block         [H, BS, D]
        k_i = kb[:, cols].reshape(H, -1, D)            # copy of the kept key blocks [H, r*BS, D]
        v_i = vb[:, cols].reshape(H, -1, D)
        w = torch.softmax((q_i @ k_i.transpose(-1, -2)) * scale, dim=-1)   # [H, BS, r*BS]
        out[:, i * BS:(i + 1) * BS] = w @ v_i
    return out


def sparse_v2_views(q, k, v, plan: SparsePlan):
    """v2 (final): identical maths, but when the kept blocks are next to each other (always
    true for a sliding window) we slice K and V in place instead of copying them."""
    H, N, D = q.shape
    BS, scale = plan.BS, 1.0 / math.sqrt(D)
    out = torch.empty_like(q)
    for i, runs in enumerate(plan.runs):
        q_i = q[:, i * BS:(i + 1) * BS]
        if len(runs) == 1:                             # one run of neighbours: a view, no copy
            a, b = runs[0]
            k_i, v_i = k[:, a * BS:b * BS], v[:, a * BS:b * BS]
        else:                                          # several runs: join just those pieces
            k_i = torch.cat([k[:, a * BS:b * BS] for a, b in runs], dim=1)
            v_i = torch.cat([v[:, a * BS:b * BS] for a, b in runs], dim=1)
        w = torch.softmax((q_i @ k_i.transpose(-1, -2)) * scale, dim=-1)
        out[:, i * BS:(i + 1) * BS] = w @ v_i
    return out


def sparse_padded_batched(q, k, v, plan: SparsePlan):
    """Alternative we tried: no Python loop - gather every query block's key blocks at once,
    padding short rows to the longest row, and do ONE batched matrix multiply.
    Fast when all rows are equal, but the padding is wasted work when they are not."""
    H, N, D = q.shape
    BS, nb, R, scale = plan.BS, plan.nb, plan.R, 1.0 / math.sqrt(D)
    qb = q.view(H, nb, BS, D)
    kg = k.view(H, nb, BS, D)[:, plan.pad_idx].reshape(H, nb, R * BS, D)
    vg = v.view(H, nb, BS, D)[:, plan.pad_idx].reshape(H, nb, R * BS, D)
    s = (qb @ kg.transpose(-1, -2)) * scale                                # [H, nb, BS, R*BS]
    s = s.masked_fill(~plan.pad_valid, float("-inf"))                      # ignore padding
    return (torch.softmax(s, dim=-1) @ vg).reshape(H, N, D)


block_sparse_attention = sparse_v2_views          # the final kernel
SPARSE_VERSIONS = {"v1_gather": sparse_v1_gather, "v2_views": sparse_v2_views,
                   "padded_batched": sparse_padded_batched}


# ============================================================ simple cost models
def dense_flops(H, N, D):
    """Q·Kᵀ is 2·N·N·D operations, weights·V another 2·N·N·D, per head."""
    return 4.0 * H * N * N * D


def sparse_flops(H, computed_blocks, BS, D):
    """Same formula, but only for the blocks we compute (each block is BS x BS)."""
    return 4.0 * H * computed_blocks * BS * BS * D


def dense_score_bytes(H, N):
    """The N x N score matrix in float32 (4 bytes per number), for every head."""
    return 4 * H * N * N
