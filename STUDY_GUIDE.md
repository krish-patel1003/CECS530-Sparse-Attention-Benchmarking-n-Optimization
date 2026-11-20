# Study guide: Sparse Attention Benchmarking and Optimization

Read this top to bottom once. Every idea in the project is here in plain words,
with the numbers you need and the questions you are likely to be asked.

---

## 1. The project in 30 seconds

> Attention compares every token with every other token, so its cost grows with N².
> We cut the attention matrix into 64×64 blocks and **only compute the blocks a sparsity
> pattern keeps**. We implemented this in PyTorch, built a benchmarking framework, and
> measured time, throughput, memory, computation and speed-up for different sequence
> lengths, sparsity levels and patterns, on the CPU and the GPU of an Apple M4 laptop.
> Main result: at 8,192 tokens our block-sparse attention is **14.7× faster** than dense
> attention and needs **about 5 MB instead of 4 GB** of extra memory.

---

## 2. Attention, in plain words

Each token (word piece) is turned into three lists of 64 numbers:
- **Q (query)** – what this token is looking for
- **K (key)** – what this token offers
- **V (value)** – the information this token passes on

Attention does three steps:
1. **Score** every pair of tokens: `scores = Q · Kᵀ` → a table of **N × N** numbers.
2. **Softmax** each row: turns scores into weights that add up to 1.
3. **Mix**: output = weights · V (a weighted average of the values).

Formula: `output = softmax(Q·Kᵀ / √D) · V`  (D = 64; dividing by √D just keeps numbers in a sensible range).

**Why it is expensive:** the score table has N × N entries.
Double N → **4× the work and 4× the memory**.
At 16,384 tokens and 8 heads the table is 8 GB (and softmax makes a second one of the same size).

---

## 3. Sparse attention and blocks

**Idea:** most tokens only need some other tokens (their neighbours plus a few important ones).
So we skip most of the table.

**Blocks:** we cut the N×N table into tiles of **64 × 64**. A tile is either computed
completely or skipped completely. Why blocks instead of single entries? A 64×64 tile is a
small *dense* matrix multiplication, which hardware does efficiently. Skipping single
entries would be scattered and slow.

**Density** = fraction of blocks computed. 12.5% density = 1 block in 8.
**Ideal speed-up** = 1 / density (12.5% → 8×).
**Computational reduction** = dense FLOPs ÷ sparse FLOPs (same number as the ideal speed-up).

### The 5 patterns (configurable in `sparse_attention/patterns.py`)
| Pattern | Picture | Used by |
|---|---|---|
| dense | every block | normal attention (baseline) |
| local | a band along the diagonal: each block sees its neighbours | Longformer (local part), Mistral |
| strided | band + every s-th column ("summary" blocks) | Sparse Transformer (OpenAI) |
| longformer | band + a few **global** blocks that see everything and are seen by everything | Longformer / BigBird |
| random | the same number of random blocks in every row | test pattern |

`make_pattern(name, N, BS, target_density=...)` picks the pattern's size so that the
density is as close as possible to what you ask for. That is how we sweep sparsity.

---

## 4. Our implementation (what to say when asked "what did you implement?")

File: `sparse_attention/attention.py`. Everything is PyTorch tensor code.

**Dense attention (baseline):** the three steps above, building the full N×N table.

**Masked dense (the naive way to do sparse):** compute the full table, then set the skipped
entries to −∞ before softmax. Correct answer, but **no work is saved** – it is actually a
bit slower than dense because masking is extra work.

**Block-sparse attention – our kernel:**
```
for each query block i (64 queries):
    take only the key blocks and value blocks that block i needs   (from the pattern)
    scores  = q_i · k_iᵀ        (small: 64 × (kept blocks × 64))
    weights = softmax(scores)
    out_i   = weights · v_i
```
Skipped blocks are never touched, so work and memory shrink with the density.
The result is **exactly the same** as masked dense attention (we test this).

**The "plan":** before running, we turn the pattern into a list per query block: "block i
needs key blocks [3, 4, 5, 9]". This is computed once and reused (a model reuses its
pattern for every input), so it is not timed.

### Optimizations we made (and measured)
| Version | What it does | Result (CPU / GPU, N=4096, local 12.5%) |
|---|---|---|
| masked dense | compute everything, then mask | 242 ms / 65 ms – no saving |
| **v1 gather** | per query block, **copy** the kept K/V blocks into one tensor, then attention | 32 ms / 40 ms |
| **v2 views (final)** | if the kept blocks are next to each other (always true for a window), **use them in place without copying** | **25 ms / 15 ms** |
| padded batch (rejected) | do all query blocks in ONE big multiply by padding every row to the longest row | 39 ms; **317 ms** on Longformer |

How to explain each:
- **Skipping blocks** is the main optimization: 242 ms → 32 ms.
- **v2 "views":** copying data costs time and memory traffic. In PyTorch, a *slice* of
  consecutive rows (`k[:, a:b]`) is a "view": it points at the existing memory, no copy.
  For a sliding window the needed blocks are consecutive, so we slice instead of copy.
  22% faster on CPU, **2.7× faster on the GPU** (the GPU is very sensitive to extra copies).
  For random patterns the blocks are scattered, so there is less to gain (no change on CPU).
- **Padded batch – why rejected:** one big multiply sounds faster, but it has to give every
  row the same length. In Longformer, the global row needs ALL 64 blocks while normal rows need
  ~9, so every row gets padded to 64 → we compute almost the whole dense matrix (317 ms).
  At 16k tokens it would need 24 GB. Lesson: **uneven rows + padding = wasted work**.
- **Block size** (another tuning choice): see section 6.

---

## 5. The benchmarking framework

`run_experiments.py` – one command runs everything, each experiment writes a CSV
(`results/`) and a log (`results/logs/`); `plot_results.py` draws every graph (`figures/`).

Configurable: `--device cpu|mps`, `--heads`, `--dim`, `--bs` (block size); patterns and
densities inside each experiment.

**How we measure:**
- **Time:** run once to warm up, then 7 times; we report the **median** (middle value, so one
  unlucky slow run does not distort it). On the GPU we wait for the GPU to finish
  (`torch.mps.synchronize()`) before stopping the clock.
- **Throughput:** tokens per second = N ÷ time.
- **Computation:** FLOPs = 4·H·N²·D for dense; 4·H·(computed blocks)·64²·D for sparse.
  (Q·Kᵀ is 2·N·N·D operations, weights·V another 2·N·N·D, per head.)
- **Memory:** we run one call in a **fresh process** and ask the operating system for the
  peak memory before and after. The difference is what the call needed.
- **Correctness:** every version is compared with a high-precision (float64) reference.
  **168 / 168 checks pass**, largest error ≈ 1e-6 (normal rounding for float32).

Setup: Apple M4 (4 performance + 6 efficiency CPU cores, 10-core GPU, 16 GB),
PyTorch 2.11, 8 heads, D = 64, block size 64, float32.

---

## 6. Every graph, explained

### `fig_seqlen` – time and speed-up vs sequence length (CPU)
- Both axes in (a) are log scale, so a straight line = a power law. **Dense has slope 2
  (quadratic); local window and Longformer have slope 1 (linear)**, because each query block
  always sees the same number of blocks.
- 8,192 tokens: dense 764 ms, **local 52 ms → 14.7×**. The FLOP reduction is 14.5× – the
  speed-up matches the work saved almost exactly.
- 16,384 tokens: dense does not run (two 8 GB matrices); local takes **104 ms**, PyTorch's
  built-in attention 1,378 ms (**13× slower** than ours).
- Short sequences (512): the window covers 81% of the matrix anyway, so nothing to skip –
  sparse is slightly slower than dense.
- Random 12.5% keeps a **fixed fraction**, so it stays quadratic; it saves 8× the work but
  gets 5.7× (scattered blocks must be copied).

### `fig_flops_throughput` – throughput and computation
- (a) Throughput of the local window stays ~157,000 tokens/s at every length; dense drops
  from 240,000 to 10,700 tokens/s.
- (b) FLOPs: dense grows 4× per doubling; local/Longformer 2× per doubling. At 16k tokens:
  550 GFLOP vs 19 GFLOP → **29× less computation**.

### `fig_sparsity` – speed-up vs density (CPU, N = 4096)
- Dotted line = ideal (1/density). Our lines follow it.
- At 12.5% density: **5.1–6.0× speed-up (64–81% of ideal)**. At 3%: **~12×**.
- **Masked dense (pink) stays at ~0.65× at every density** – masking never helps.
- At 100% density sparse is 0.77× of dense: doing 64 small multiplies is less efficient than
  one big one (per-block overhead). That is the price of the flexibility.
- Why not 100% of ideal? Fixed costs per block (Python loop step, starting each small
  multiply) do not shrink when the work shrinks, and scattered patterns need copies.
- (b) "% of ideal": local is best (no copies thanks to v2), random/strided lowest.

### `fig_patterns` – the patterns and their time
- Top: pictures of which blocks are computed. Bottom: all patterns at ≈12.5% density run
  in 28–33 ms vs 170 ms dense. **What matters is how many blocks, not the shape.** Local is
  fastest (no copying).

### `fig_memory` – measured memory (CPU)
- Dense and masked dense: **4,099 MB at 8,192 tokens**, matching our prediction of two N×N
  float32 matrices (scores + weights) = 2·8·N²·4 bytes = 4,096 MB.
- Block-sparse: **about 5 MB at every length** – it only ever holds one query block's small
  score table. **≈ 760× less at 8k tokens** (4,099 MB vs 5.4 MB).
- PyTorch built-in: ~3–4 MB (it uses a smarter fused algorithm; reference only).

### `fig_ablation` – the implementation versions (CPU and GPU)
- See the table in section 4. Key messages: skipping blocks gives the big win; views (v2)
  help most on the GPU; padding is a disaster on uneven patterns (Longformer).

### `fig_blocksize` – block size trade-off (±256-token window, N = 4096)
- Big blocks cover entries outside the window: useful work falls from **97% (16) to 81% (128)**.
- But big blocks mean fewer, bigger multiplies, which hardware does more efficiently:
  time falls from **51 ms (16) to 23 ms (128)**.
- Trade-off: precision vs efficiency. We use 64 as a balance (89% useful, 25 ms).

### `fig_threads` – CPU threads
- PyTorch uses 4 threads by default = the M4's 4 **performance cores**.
- 4 threads: dense 2.4×, sparse 2.0× faster than 1 thread. More threads do not help:
  the extra 6 cores are slower **efficiency cores**, and part of the work is moving data (likely
  limited by memory speed rather than core count).

### `fig_cpu_vs_gpu`, `fig_seqlen_mps`, `fig_sparsity_mps` – the Apple GPU
- **Same code**, just `--device mps`.
- GPU at 8,192 tokens: dense 296 ms → **local 22 ms (13.2×)**.
- GPU at 16,384: PyTorch's built-in attention **runs out of memory** (it asks for one 8 GB
  buffer); ours runs in **34 ms** (3× faster than our CPU run).
- GPU sparsity sweep: speed-up is lower than on the CPU and **stops improving below ~12%
  density (time floor ≈ 10–15 ms)**. Reason: each small GPU operation has a fixed start-up
  cost ("kernel launch overhead"); with 64 query blocks × a few operations each, that fixed
  cost dominates once the real work is tiny. The GPU is also extremely good at one big dense
  multiply, so sparse only wins below ~25% density there. High-density GPU timings vary
  ±40% between runs.

---

## 7. Computer-architecture take-aways (the "analysis" part)
1. **Work vs memory:** dense attention's memory grows with N² and hits the 16 GB limit;
   block-sparse memory stays tiny because only one small tile exists at a time.
2. **Memory traffic matters:** avoiding copies (views) gave up to 2.7× on the GPU – moving
   data costs as much as computing on it.
3. **Big operations are more efficient than many small ones:** that is why sparse at 100%
   density is slower than dense, why bigger blocks run faster, and why the GPU has a time
   floor. This is "per-operation overhead".
4. **Load balance / padding:** uneven rows waste work when you force them into one shape.
5. **Heterogeneous cores:** 4 fast + 6 slow cores; more threads ≠ faster.

---

## 8. Likely questions and short answers

**Q: What exactly did you implement?**
A block-sparse attention kernel in PyTorch: for each block of 64 queries we gather only the
key/value blocks the pattern keeps and run attention on that piece. Plus dense and masked
baselines, five configurable patterns, and a benchmarking framework.

**Q: Isn't PyTorch doing the work?**
PyTorch does the arithmetic (matrix multiply, softmax), like any library. Our code decides
which blocks to compute, how they are fetched (views vs copies), and how they are grouped –
that is where the savings come from (242 ms → 25 ms vs masked dense, which uses the same
PyTorch operations).

**Q: Why blocks of 64?**
Balance: blocks small enough to follow the pattern (89% useful work) but big enough that each
multiply is efficient. Our block-size experiment shows the trade-off.

**Q: Is the result exactly the same as dense attention?**
Yes, for the entries the pattern keeps – same as masked dense. 168 checks against a float64
reference, max error ≈ 1e-6 (float32 rounding).

**Q: Why isn't the speed-up equal to the ideal 1/density?**
Fixed costs per block (loop step, starting each small multiply) don't shrink with the work;
scattered patterns need data copies; small multiplies are less efficient than one big one.

**Q: Why is masked dense slower than dense?**
It does all the dense work plus the masking step.

**Q: How did you measure memory?**
One call per fresh process; we read the operating system's peak memory before and after.
Dense matched our formula (2 · heads · N² · 4 bytes) within 0.1%.

**Q: Why median time?**
It ignores occasional slow runs caused by background activity on the laptop.

**Q: Why does PyTorch's built-in attention use so little memory but run out of memory on the GPU?**
On the CPU it uses a fused algorithm that never stores the whole table. On the Apple GPU
(at the version we used) it allocated a full N×N buffer – 8 GB at 16k tokens – and failed.

**Q: Why does the GPU stop getting faster at low density?**
Each GPU operation has a fixed launch cost. Our loop launches a few operations per query
block, so at very low density those fixed costs (~10–15 ms total) dominate.

**Q: Why not one big batched multiply?**
We tried it (padded batch). It must pad every row to the longest row; for Longformer the
global row needs all blocks, so everything is padded to full size – 10× slower, 24 GB at 16k.

**Q: Why doesn't using 10 threads help?**
PyTorch uses the 4 fast performance cores by default. The other 6 are slower efficiency
cores; adding them makes threads wait for the slowest one, and part of the work is moving data,
which more cores don't speed up. (We measured: no gain beyond 4 threads.)

**Q: What's the difference between local, Longformer and BigBird?**
Local: only neighbours. Longformer: neighbours + a few global tokens. BigBird: neighbours +
global + random blocks. (Our "longformer" pattern = window + global blocks.)

**Q: What would you do next?**
Remove the remaining Python loop with a custom GPU kernel (e.g. Triton/Metal) to remove the
per-block overhead, use 16-bit numbers, and support training (backward pass).

**Q: Where is the trade-off between performance and memory?**
Sparse saves both time and memory, but you give up attention to the skipped blocks (accuracy
of the model is outside our scope), and at high density or short sequences dense is faster.
Bigger blocks are faster but waste work; padding is fast for uniform patterns but wastes
memory for uneven ones.

---

## 9. Files
```
sparse_attention/patterns.py   the 5 patterns + density targeting
sparse_attention/attention.py  dense, masked dense, block-sparse v1/v2, padded batch, FLOP formulas
run_experiments.py             the benchmarking framework (7 experiments, CPU or GPU)
memory_probe.py                one call in a fresh process -> peak memory
plot_results.py                all graphs
tests/test_correctness.py      168 correctness checks
results/  figures/  report/
```
