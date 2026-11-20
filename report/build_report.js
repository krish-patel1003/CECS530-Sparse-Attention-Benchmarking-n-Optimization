// Build report/Sparse_Attention_Report.docx
//   NODE_PATH=$(npm root -g) node report/build_report.js
const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, AlignmentType, Footer, PageNumber, TableOfContents, PageBreak } = require("docx");
const { P, H1, H2, H3, B, N, Fig, Tbl, Code, styles, numbering, PAGE_W, MARGIN } = require("./lib");

const title = [
  new Paragraph({ spacing: { before: 1800 }, children: [] }),
  new Paragraph({ style: "Title", children: [new TextRun("Sparse Attention Benchmarking and Optimization")] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 },
    children: [new TextRun({ text: "A block-sparse attention kernel in PyTorch and a benchmarking framework, evaluated on the CPU and GPU of an Apple M4", size: 26, color: "404040" })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Computer Architecture — Term Project Final Report", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Team members: Krish Patel, Aarsh Sanjay Sharma", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Fall 2026", size: 24 })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", bold: true, size: 30, color: "1F3864" })], spacing: { after: 160 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
  new Paragraph({ children: [new PageBreak()] }),
];

const F = (f) => `figures/${f}.png`;

const body = [
  // ================================================================ 1
  H1("1. Introduction"),
  P("Attention is the core operation of Transformer models. Every token is compared with every other token: for a sequence of N tokens " +
    "and head size D it computes output = softmax(Q·Kᵀ/√D)·V, where Q, K and V hold a query, key and value vector for each token. " +
    "The comparison step produces an **N × N table of scores**, so the computation and the memory both grow with **N²**: doubling the " +
    "sequence length makes attention four times more expensive. At 16,384 tokens and 8 heads, one forward pass needs about 550 billion " +
    "floating-point operations, and the score table alone needs 8 GB, half of the 16 GB of memory of the laptop used in this project."),
  P("**Sparse attention** reduces this cost by letting each token look at only some other tokens, for example its neighbours plus a few " +
    "important tokens. In **block-sparse** attention the N × N table is cut into blocks (here 64 × 64 tokens) and only the blocks chosen by a " +
    "sparsity pattern are computed. In theory the work drops in proportion to the fraction of blocks kept (the *density*)."),
  P("**Objective.** Investigate how much computation and memory block-sparse attention really saves compared with standard dense " +
    "attention, and why. Concretely we:"),
  B("implemented a block-sparse attention kernel (with dense and masked-dense baselines) in PyTorch, with five configurable sparsity patterns and any sequence length;"),
  B("built a benchmarking framework that measures execution time, throughput, memory usage, computation (FLOPs), computational reduction and speed-up;"),
  B("ran experiments over sequence lengths (512 to 16,384 tokens), sparsity levels (100% to 3% of blocks), patterns, block sizes, thread counts, and on both the CPU and the GPU;"),
  B("analysed the results from a computer-architecture point of view (memory capacity, memory traffic, per-operation overhead, core types)."),
  P("Hardware: Apple M4 laptop (4 performance + 6 efficiency CPU cores, 10-core GPU, 16 GB unified memory). Software: Python 3.12, PyTorch 2.11. " +
    "Default configuration: batch 1, 8 heads, D = 64, block size 64, float32."),

  // ================================================================ 2
  H1("2. Related Work and Our Contribution"),
  H2("2.1 Related work"),
  P("**Sparse attention patterns.** The Sparse Transformer (Child et al., 2019) introduced *strided* patterns; Longformer (Beltagy et al., 2020) " +
    "uses a *sliding window* plus a few *global* tokens; BigBird (Zaheer et al., 2020) adds *random* blocks to window and global attention. " +
    "Mistral-7B uses sliding-window attention in a production model."),
  P("**Efficient implementations.** OpenAI's blocksparse library and DeepSpeed Sparse Attention provide block-sparse GPU kernels; " +
    "FlashAttention (Dao et al., 2022) computes dense attention in tiles so the N × N table is never stored; PyTorch provides a built-in fused " +
    "attention (scaled_dot_product_attention) and FlexAttention for custom masks. Most of this work targets NVIDIA GPUs."),
  H2("2.2 What we did differently"),
  B("**Our own block-sparse kernel**, written so that each step is visible: for each block of queries, fetch only the key/value blocks the " +
    "pattern keeps and run ordinary attention on that piece. Skipped blocks are never touched."),
  B("**Two optimizations we measured**: (1) skipping blocks instead of masking them, and (2) using *views* of neighbouring blocks instead of " +
    "copying them (v2). We also tried, measured and rejected a padded batched design, which explains a load-imbalance problem."),
  B("**A benchmarking framework** with five patterns (dense, local, strided, Longformer, random), density targeting, any sequence length, " +
    "CPU or GPU, process-isolated memory measurement and correctness checks."),
  B("**Evaluation on Apple silicon** (CPU and GPU with the same code), a platform the work above does not target, including a comparison " +
    "with PyTorch's built-in attention."),

  // ================================================================ 3
  H1("3. Methodology / Project Implementation"),
  H2("3.1 Steps we followed"),
  N("Implemented dense attention and a masked-dense baseline, and the five sparsity patterns."),
  N("Implemented the block-sparse kernel (v1), then optimized it (v2), and tried a padded batched variant."),
  N("Wrote correctness tests against a float64 reference (168 checks, CPU and GPU)."),
  N("Built the benchmarking framework and memory probe, ran seven experiments on the CPU and three on the GPU."),
  N("Plotted the results and analysed them."),
  H2("3.2 Sparsity patterns (configurable)"),
  P("A pattern is a True/False grid of size (N/64) × (N/64): entry (i, j) says whether query block i looks at key block j. " +
    "File `sparse_attention/patterns.py` implements five patterns. `make_pattern(name, N, block_size, target_density)` chooses the pattern's " +
    "size (e.g. the window width) to get as close as possible to a requested density, which lets every experiment sweep the sparsity level."),
  ...Tbl(["Pattern", "What each query block sees", "Used by"], [
    ["dense", "every block", "standard attention (baseline)"],
    ["local", "its neighbours (a band along the diagonal)", "Longformer (local part), Mistral"],
    ["strided", "neighbours + every s-th block", "Sparse Transformer"],
    ["longformer", "neighbours + a few global blocks (they see all, all see them)", "Longformer, BigBird"],
    ["random", "the same number of random blocks in each row", "test pattern"],
  ], [1.2, 3.2, 2.2], "Sparsity patterns supported by the framework."),
  ...Fig(F("fig_patterns"), "The five patterns (top, shown at 25% density) and their execution time at ≈12.5% density for N = 4,096 on the CPU (bottom)."),
  H2("3.3 The attention implementations"),
  ...Tbl(["Implementation", "What it does", "Role"], [
    ["Dense", "Q·Kᵀ → softmax → ·V, builds the full N × N table", "baseline"],
    ["Masked dense", "dense, then hides skipped entries with −∞ before softmax", "naive sparse baseline"],
    ["Block-sparse v1 (gather)", "per query block: copy the kept K/V blocks together, run attention on them", "our kernel, first version"],
    ["Block-sparse v2 (views)", "same, but neighbouring blocks are used in place (no copy)", "our final kernel"],
    ["Padded batch", "all query blocks in one multiply, every row padded to the longest row", "tried and rejected"],
    ["PyTorch built-in", "torch's scaled_dot_product_attention", "reference only"],
  ], [1.8, 3.6, 1.6], "Implementations compared."),
  P("The block-sparse kernel (`sparse_attention/attention.py`) in pseudocode:"),
  ...Code([
    "for each query block i (64 queries):",
    "    k_i, v_i = the key/value blocks that block i needs       # from the pattern",
    "    scores   = q_i @ k_iᵀ / sqrt(D)                          # 64 × (kept blocks × 64)",
    "    weights  = softmax(scores)",
    "    out_i    = weights @ v_i",
  ]),
  P("Because only the kept blocks are fetched and multiplied, the work is proportional to the number of kept blocks, and the extra memory is " +
    "only one query block's small score table at a time. The result is identical to masked dense attention. Before running, the pattern is " +
    "turned into a *plan*: the list of key blocks for every query block (and, for v2, which of them are consecutive). The plan is built once " +
    "per pattern and reused, as a model reuses its pattern for every input, so it is not part of the timing."),
  H3("Optimization 1: skip, don't mask"),
  P("Masked dense attention computes all N² scores and throws most away. The block-sparse kernel never computes the skipped blocks. This is the main source of savings."),
  H3("Optimization 2: views instead of copies (v2)"),
  P("In v1, the kept key/value blocks of a row are *copied* into one tensor before multiplying. Copying costs time and memory traffic. In PyTorch, " +
    "a slice of consecutive rows (`k[:, a:b]`) is a *view*: it points at the existing memory and copies nothing. For a sliding window the kept blocks " +
    "of a row are always consecutive, so v2 slices instead of copying (scattered blocks are still joined)."),
  H3("Tried and rejected: one padded batched multiply"),
  P("Instead of a loop over query blocks, all query blocks can be processed by one large batched multiply. That needs every row to have the same " +
    "number of blocks, so short rows are padded to the longest row. For uneven patterns such as Longformer, where a global row needs all blocks, " +
    "almost everything is padded to full size."),
  H2("3.4 Benchmarking framework"),
  P("`run_experiments.py` runs every experiment with one command and writes a CSV file and a console log per experiment; `plot_results.py` draws " +
    "every graph. Options: `--device cpu|mps`, `--heads`, `--dim`, `--bs` (block size)."),
  B("**Time**: one warm-up run, then 7 runs; we report the **median**. On the GPU the clock is stopped only after the GPU has finished (`torch.mps.synchronize`)."),
  B("**Throughput**: tokens per second = N ÷ time."),
  B("**Computation**: FLOPs = 4·H·N²·D for dense attention (Q·Kᵀ and weights·V each cost 2·N·N·D per head) and 4·H·(computed blocks)·64²·D for block-sparse. **Computational reduction** = dense FLOPs ÷ sparse FLOPs."),
  B("**Speed-up** = dense time ÷ sparse time, compared with the **ideal speed-up** 1 ÷ density."),
  B("**Memory**: each case runs once in a fresh process; the operating system's peak memory before and after the call gives the extra memory it needed."),
  B("**Correctness**: every implementation is compared with a float64 reference."),
  ...Tbl(["Experiment", "What varies", "Measures"], [
    ["correctness", "4 shapes × 5 patterns, CPU and GPU", "max error vs float64 reference"],
    ["seqlen", "N = 512 … 16,384", "time, throughput, FLOPs, speed-up"],
    ["sparsity", "density 100% → 3%, 4 patterns + masked dense", "speed-up vs ideal"],
    ["ablation", "masked dense, v1, v2, padded batch", "effect of each design choice"],
    ["blocksize", "block size 16 … 128", "useful work vs speed"],
    ["threads", "1 … 10 CPU threads", "parallel scaling"],
    ["memory", "N = 1,024 … 16,384", "measured peak memory"],
  ], [1.4, 2.8, 2.6], "Experiments (seqlen, sparsity and ablation were also run on the Apple GPU)."),

  // ================================================================ 4
  H1("4. Results"),
  H2("4.1 Correctness"),
  P("All **168 checks pass** on both CPU and GPU (tolerance 1e-4). The largest error is 1.2e-6, which is ordinary float32 rounding. The block-sparse " +
    "versions give exactly the same result as masked dense attention."),
  ...Fig(F("term_correctness"), "Console output of the correctness experiment.", 4.4),

  H2("4.2 Execution time and throughput vs sequence length (CPU)"),
  P("Three sparse configurations were compared with dense attention: a local window of ±4 blocks and Longformer (window + 1 global block) both keep " +
    "a *fixed number of blocks per row*, so their density falls as N grows; random keeps a *fixed 12.5%* of blocks."),
  ...Tbl(["N", "Dense", "PyTorch built-in", "Sparse local", "Sparse Longformer", "Sparse random 12.5%", "Local speed-up", "Work reduction (local)"], [
    ["512", "2.1", "1.1", "2.5", "2.8", "1.0", "0.8×", "1.2×"],
    ["1,024", "10.0", "3.8", "5.7", "6.6", "3.1", "1.7×", "2.1×"],
    ["2,048", "39.8", "14.2", "12.5", "15.0", "9.9", "3.2×", "3.8×"],
    ["4,096", "167.3", "82.9", "25.9", "36.5", "38.8", "6.5×", "7.4×"],
    ["8,192", "763.7", "344.0", "52.0", "62.5", "134.7", "14.7×", "14.5×"],
    ["16,384", "out of memory*", "1,377.7", "104.2", "130.1", "625.9", "—", "28.7×"],
  ], [0.8, 0.9, 1.1, 1, 1.1, 1.2, 1, 1.2], "CPU time per forward pass (ms). *Dense needs two 8 GB N×N matrices at 16,384 tokens and was not run."),
  P("On log-log axes (Figure 2a) the dense line has slope 2 (quadratic) while local and Longformer have slope 1 (linear): from 8,192 to 16,384 tokens the " +
    "local window time only doubles (52 → 104 ms). At 8,192 tokens the local window is **14.7× faster** than dense, matching the **14.5× reduction in " +
    "work**. At 16,384 tokens it is **13× faster than PyTorch's built-in attention**. At 512 tokens the window covers 81% of the matrix, so there is " +
    "almost nothing to skip and dense is slightly faster. Throughput (Figure 3a) stays at about 157,000 tokens/s for the local window at every length, " +
    "while dense falls from 240,000 to 10,700 tokens/s."),
  ...Fig(F("fig_seqlen"), "CPU: (a) time per forward pass vs sequence length (log-log), (b) speed-up of block-sparse over dense."),
  ...Fig(F("fig_flops_throughput"), "(a) Throughput in tokens per second, (b) computation per forward pass: the computational reduction."),
  ...Fig(F("term_seqlen"), "Console output of the sequence-length experiment.", 6.2),

  H2("4.3 Speed-up vs sparsity level (CPU, N = 4,096)"),
  ...Tbl(["Density", "local", "strided", "longformer", "random", "masked dense", "ideal"], [
    ["≈100%", "0.77×", "—", "0.77×", "0.77×", "0.73×", "1×"],
    ["≈50%", "1.88×", "1.19×", "1.44×", "1.15×", "0.65×", "2×"],
    ["≈25%", "4.24×", "2.90×", "2.84×", "2.53×", "0.63×", "4×"],
    ["≈12.5%", "5.98×", "5.51×", "5.52×", "5.12×", "0.66×", "8×"],
    ["≈6%", "9.22×", "9.14×", "8.02×", "8.79×", "0.65×", "16×"],
    ["≈3–5%", "12.39×", "10.61×", "8.41×", "12.21×", "0.69×", "21–32×"],
  ], [1, 1, 1, 1, 1, 1.2, 0.8], "Speed-up over dense attention (170 ms) at N = 4,096 on the CPU, by pattern and density."),
  P("The speed-up follows the ideal 1/density line (Figure 4): at 12.5% density the kernel is **5.1–6.0× faster** (64–81% of the ideal 8×), at about " +
    "3% density **about 12× faster**. **Masked dense is never faster than dense (≈0.65×) at any density**: computing all scores and then masking saves nothing " +
    "and adds the masking step. At 100% density the block-sparse kernel is 0.77× of dense, the cost of doing 64 smaller multiplies instead of one large one."),
  ...Fig(F("fig_sparsity"), "CPU, N = 4,096: (a) speed-up vs density against the ideal and masked dense, (b) percentage of the ideal speed-up reached."),
  ...Fig(F("term_sparsity"), "Console output of the sparsity experiment.", 6.0),

  H2("4.4 Implementation versions (ablation)"),
  ...Tbl(["Version", "CPU local", "CPU longformer", "CPU random", "GPU local", "GPU longformer", "GPU random"], [
    ["masked dense", "241.9", "246.1", "244.6", "65.2", "68.3", "61.1"],
    ["v1 gather", "32.3", "33.7", "31.3", "40.4", "34.8", "27.1"],
    ["**v2 views (final)**", "**25.4**", "**29.9**", "32.2", "**15.1**", "**18.0**", "**17.1**"],
    ["padded batch", "39.4", "317.4", "33.6", "41.1", "273.9", "32.1"],
  ], [1.6, 1, 1.1, 1, 1, 1.1, 1], "Time (ms) of each version at N = 4,096, ≈12.5% density."),
  P("Skipping blocks gives the largest gain (242 → 32 ms on the CPU). Using views (v2) is **22% faster than v1 on the CPU** for the local window and " +
    "**2.7× faster on the GPU**; at 16,384 tokens v2 is 1.3× (CPU) and 3.0× (GPU) faster than v1. For random patterns the blocks are scattered, so v2 " +
    "must still join pieces and gains nothing on the CPU. The **padded batch is 10× slower on Longformer** (317 vs 30 ms) because the global row needs " +
    "all 64 blocks and every row is padded to 64 blocks; at 16,384 tokens it would need 24 GB and was skipped."),
  ...Fig(F("fig_ablation"), "Time of each implementation version on the CPU (left) and GPU (right)."),
  ...Fig(F("term_ablation"), "Console output of the ablation experiment (CPU).", 5.2),

  H2("4.5 Block size"),
  ...Tbl(["Block size", "Blocks computed", "Useful work", "Time (ms)", "GFLOP/s"], [
    ["16", "12.5%", "97%", "51.0", "84"], ["32", "12.8%", "94%", "32.4", "136"],
    ["64", "13.6%", "89%", "25.2", "185"], ["128", "15.0%", "81%", "23.0", "224"],
  ], [1, 1.2, 1, 1, 1], "Block-size trade-off for a ±256-token sliding window, N = 4,096, CPU."),
  P("Large blocks follow a token-level pattern less precisely: with 128-token blocks only 81% of the computed entries are actually needed. But " +
    "large blocks mean fewer, larger multiplies, which run more efficiently (84 → 224 GFLOP/s), so the time still falls. Block size 64 is a balance " +
    "between useful work (89%) and speed."),
  ...Fig(F("fig_blocksize"), "(a) Fraction of computed entries that are actually needed, (b) execution time, by block size."),

  H2("4.6 CPU threads"),
  P("PyTorch uses 4 threads by default, one per performance core. With 4 threads dense attention runs 2.4× and block-sparse 2.0× faster than " +
    "with 1 thread; 6, 8 or 10 threads give no further gain, because the remaining 6 cores are slower efficiency cores."),
  ...Fig(F("fig_threads"), "Speed-up vs number of CPU threads (N = 4,096). The shaded region uses efficiency cores.", 5.0),

  H2("4.7 Memory usage"),
  ...Tbl(["N", "Dense", "Masked dense", "PyTorch built-in", "Block-sparse (ours)", "Prediction for dense"], [
    ["1,024", "65.6", "66.0", "3.2", "5.3", "64"], ["2,048", "258.4", "258.6", "3.2", "5.3", "256"],
    ["4,096", "1,026.4", "1,042.6", "3.3", "5.3", "1,024"], ["8,192", "4,099.2", "4,162.6", "4.1", "5.4", "4,096"],
    ["16,384", "not run", "not run", "4.0", "4.7", "16,384"],
  ], [0.9, 1, 1.1, 1.2, 1.3, 1.3], "Measured peak extra memory (MB) of one call, CPU."),
  P("Dense and masked dense memory grows with N² and matches our prediction of two N × N float32 matrices (scores and weights), " +
    "2 × 8 × N² × 4 bytes, within 0.1% at 8,192 tokens: **4,099 MB**. Block-sparse attention needs **about 5 MB at every length**, about **760× less** at 8,192 " +
    "tokens, because only one query block's small score table exists at a time. Masking saves no memory at all."),
  ...Fig(F("fig_memory"), "Measured peak extra memory of one attention call vs sequence length (CPU).", 5.6),
  ...Fig(F("term_memory"), "Console output of the memory experiment.", 5.0),

  H2("4.8 Apple GPU"),
  P("The same code runs on the GPU with `--device mps`. At 8,192 tokens, dense takes 296 ms and the local-window kernel **22 ms (13.2× faster)**. " +
    "At 16,384 tokens PyTorch's built-in attention **ran out of memory** (it requested one 8 GB buffer), while our kernel ran in **34 ms**, 3× faster " +
    "than on the CPU. In the GPU sparsity sweep the speed-up is smaller than on the CPU (3.9× for local at 12.5%) and **stops improving below about 12% " +
    "density**, where the time levels off at about 10–12 ms. At high density, sparse is slower than dense on the GPU, and these timings vary by up to ±40% between runs."),
  ...Fig(F("fig_seqlen_mps"), "Apple GPU: (a) time vs sequence length, (b) speed-up over dense."),
  ...Fig(F("fig_sparsity_mps"), "Apple GPU, N = 4,096: (a) speed-up vs density, (b) percentage of the ideal reached."),
  ...Fig(F("fig_cpu_vs_gpu"), "The same code on the CPU and the GPU: dense vs block-sparse local window.", 5.4),

  // ================================================================ 5
  H1("5. Discussion"),
  H2("5.1 Performance and memory trade-off"),
  P("Block-sparse attention reduces **both** time and memory, and the time saved tracks the work saved: at 8,192 tokens the 14.5× reduction in " +
    "FLOPs became a 14.7× speed-up. Patterns with a fixed number of blocks per row (local, Longformer) turn attention from quadratic into linear " +
    "growth, so the advantage grows with sequence length. Memory falls from gigabytes to a few megabytes, which is what allows sequences that dense " +
    "attention cannot run at all on a 16 GB machine. The trade-offs are: (1) the model attends only to the kept blocks (whether that is accurate " +
    "enough is a modelling question outside our scope); (2) for short sequences or high density, dense is faster; (3) larger blocks run faster but " +
    "waste work; (4) one big padded multiply is fast for even patterns but wastes time and memory for uneven ones."),
  H2("5.2 Why the speed-up is below the ideal (architecture view)"),
  B("**Per-operation overhead.** Each query block launches a few small operations (a step of the Python loop and small matrix multiplies). " +
    "This cost does not shrink when the work shrinks, which is why sparse at 100% density is 0.77× of dense, why bigger blocks are faster, and why " +
    "the speed-up flattens at very low density."),
  B("**Memory traffic.** Copying blocks costs time. Removing copies (v2 views) gave 22% on the CPU and 2.7× on the GPU; scattered patterns " +
    "(random, strided) still need copies and reach a lower fraction of the ideal."),
  B("**Large operations are more efficient.** One big dense multiply keeps the hardware fully busy; many small ones do not. This also explains " +
    "the block-size result (84 → 224 GFLOP/s from block size 16 to 128)."),
  B("**GPU launch overhead.** On the GPU every operation has a fixed start-up cost. With 64 query blocks × several operations, this adds up to the " +
    "≈10–12 ms floor in the GPU sparsity sweep. The GPU is also very fast at a single dense multiply, so sparse only wins there below ≈25% density."),
  B("**Load imbalance.** Uneven rows (Longformer's global row) are handled naturally by the loop, but are disastrous when forced into one padded shape."),
  B("**Core types.** The M4's 4 performance cores give a 2–2.4× speed-up; adding the 6 slower efficiency cores does not help."),
  H2("5.3 Challenges"),
  B("**Running out of memory.** Dense attention at 16,384 tokens needs two 8 GB matrices, and PyTorch's GPU attention failed with an 8 GB request; the padded " +
    "batch would have needed 24 GB. The framework records these cases instead of crashing."),
  B("**Noisy GPU timings at high density** (±40% between runs). We report medians and focus on the lower-density results, which are stable."),
  B("**A design that looked faster but was not.** The padded batched version removed the Python loop but was 10× slower on Longformer; measuring it " +
    "taught us the cost of padding uneven work."),

  // ================================================================ 6
  H1("6. Conclusion"),
  P("We implemented a block-sparse attention kernel in PyTorch, dense and masked-dense baselines, five configurable sparsity patterns, and a " +
    "benchmarking framework that measures time, throughput, memory, computation and speed-up on the CPU and GPU of an Apple M4."),
  P("Main findings:"),
  B("Sparsity only pays when skipped blocks are **never computed**: masked dense attention is never faster than dense (≈0.65×), while our kernel is " +
    "5–6× faster at 12.5% density and about 12× at 3% (CPU)."),
  B("With a sliding window, attention becomes **linear** in sequence length: **14.7× faster than dense at 8,192 tokens** and **13× faster than PyTorch's " +
    "built-in attention at 16,384 tokens** on the CPU; 13.2× faster than dense on the GPU."),
  B("Memory drops from **4,099 MB to about 5 MB** at 8,192 tokens, so sequences that dense attention cannot fit become possible."),
  B("The gap to the ideal speed-up comes from per-operation overhead, memory copies and load imbalance; avoiding copies gave up to 2.7× on the GPU."),
  P("What we learned: the savings of sparse attention depend as much on *how* the remaining work is organised (avoiding copies, avoiding padding, " +
    "choosing the block size, using the right cores) as on how much work is skipped. Next steps would be to replace the Python loop with a single " +
    "custom GPU kernel to remove per-block overhead, use 16-bit numbers, and support training."),
  H2("References"),
  B("R. Child, S. Gray, A. Radford, I. Sutskever. Generating Long Sequences with Sparse Transformers. 2019."),
  B("I. Beltagy, M. Peters, A. Cohan. Longformer: The Long-Document Transformer. 2020."),
  B("M. Zaheer et al. Big Bird: Transformers for Longer Sequences. NeurIPS 2020."),
  B("T. Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS 2022."),
  B("A. Jiang et al. Mistral 7B. 2023."),
  B("OpenAI blocksparse (2017); DeepSpeed Sparse Attention (2020); PyTorch scaled_dot_product_attention and FlexAttention documentation."),
];

const doc = new Document({
  creator: "Krish Patel, Aarsh Sanjay Sharma", title: "Sparse Attention Benchmarking and Optimization", features: { updateFields: true },
  styles, numbering,
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 15840 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: "7F7F7F" })] })] }) },
    children: [...title, ...body],
  }],
});
const out = path.join(__dirname, "Sparse_Attention_Report.docx");
Packer.toBuffer(doc).then((b) => { fs.writeFileSync(out, b); console.log("wrote", out); });
