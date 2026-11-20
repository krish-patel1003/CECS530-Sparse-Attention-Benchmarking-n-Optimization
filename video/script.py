"""Narration script for the project walkthrough video.

Each chapter = (title, [(visual key, narration), ...]).  A beat's visual lasts exactly as long
as its narration.  Numbers are spelled out for the speech engine.
"""

CHAPTERS = [
    ("Introduction", [
        ("title", "Sparse attention benchmarking and optimization. This video walks through the whole project in plain language: the problem, the idea, our code, how we measured it, and what every graph means."),
        ("roadmap", "Four parts. First, what attention is and why it gets expensive. Second, our block sparse kernel and how we improved it. Third, the benchmarking framework and every result. And last, why the results look the way they do, and what we conclude."),
    ]),
    ("What attention computes", [
        ("attn_sentence", "Take the sentence: the cat sat because it was tired. To understand the word it, a language model has to work out that it means the cat. Attention does this: every word looks at every other word, and decides how much each one matters."),
        ("qkv", "Each word is turned into three lists of sixty four numbers. A query: what am I looking for. A key: what do I offer. And a value: the information I pass on."),
        ("three_steps", "Then three steps. One: score every pair of words, by comparing one word's query with another word's key. That makes a table with N rows and N columns. Two: softmax turns each row of scores into weights that add up to one. Three: each word's output is the weighted average of the values. So it ends up mostly carrying the meaning of cat."),
        ("formula", "As a formula: softmax of Q times K transpose, over the square root of the size, times V. Code that computes this on real hardware is called an attention kernel."),
    ]),
    ("Why it gets expensive", [
        ("matrix_grow", "The problem is that table. It has N times N entries, so when the text gets twice as long, the table gets four times bigger: four times the work and four times the memory."),
        ("cost_numbers", "At sixteen thousand tokens with eight attention heads, one pass needs five hundred and fifty billion operations, and the table takes eight gigabytes. That's half the memory of the laptop we used."),
    ]),
    ("The idea: block sparsity", [
        ("tiles", "The idea is simple. Most words only need their neighbours and a few important words. So we cut the big table into blocks of sixty four by sixty four, and only compute the blocks that matter. A skipped block costs nothing at all."),
        ("patterns", "Which blocks to keep is called the pattern. Our framework has five. Dense keeps everything. Local keeps a band of neighbours. Strided adds regular summary columns. Longformer adds a few global blocks that see everything. And random keeps the same number of random blocks in each row."),
        ("density", "The fraction of blocks we keep is the density. Keep one block in eight, and the density is twelve and a half percent, so ideally the work is eight times smaller. Our question: how close does a real computer get to that ideal?"),
        ("masked_vs_skip", "One important point. A lazy way to do sparse attention is to compute the whole table, and then hide the parts you don't want. That gives the right answer, but saves nothing, because all the work was already done. Real savings only come if the skipped blocks are never computed."),
    ]),
    ("Our kernel", [
        ("kernel_loop", "Here is our kernel. For each block of sixty four queries, we look up which key blocks the pattern keeps. We fetch only those keys and values, and run ordinary attention on that small piece: scores, softmax, weighted sum. Then we move to the next query block."),
        ("kernel_code", "In code it really is five lines. We wrote it in PyTorch. PyTorch does the arithmetic, like any library would. Our code decides which blocks are computed and how they are fetched, and that is where all the savings come from."),
        ("plan", "Before running, we turn the pattern into a plan: a short list for each query block, saying which key blocks it needs. The plan is made once and reused, just like a real model reuses its pattern for every input."),
        ("exact", "Is the answer still correct? Yes. We compare every version against a high precision reference: one hundred and sixty eight checks, all passing on both the C P U and the G P U, with errors around one in a million, which is normal rounding."),
    ]),
    ("Optimizing the kernel", [
        ("copy_vs_view", "Our first version copied the needed key and value blocks into a new piece of memory before multiplying. But copying takes time. In version two we noticed that for a sliding window, the needed blocks sit right next to each other in memory. So we simply point at them, without copying. PyTorch calls that a view."),
        ("padded", "We also tried a different idea: do all query blocks in one giant multiply instead of a loop. The catch is that every row must have the same length, so short rows get padded. With Longformer, one global row needs all sixty four blocks while the others need about nine, so everything gets padded to sixty four. Most of the work is wasted."),
        ("versions_chart", "Here are the four versions, measured. Masked dense: two hundred and forty two milliseconds. Version one, which skips blocks: thirty two. Version two, which avoids copies: twenty five on the C P U, and fifteen on the G P U, which is very sensitive to copies. And the padded batch on Longformer: over three hundred. So we kept version two, and rejected padding."),
    ]),
    ("The benchmarking framework", [
        ("framework", "To measure everything, we built a benchmarking framework. One command runs every experiment, and saves a results table and a log for each. Another command draws all the graphs. Pattern, density, sequence length, block size and device are all configurable."),
        ("method", "Time is the median of seven runs after a warm up, so one slow run doesn't distort it. Throughput is tokens per second. Computation is counted with the standard formula. And memory is measured by running one call in a fresh process, and asking the operating system how much memory it peaked at. Everything ran on an Apple M4 laptop, on the C P U and the G P U."),
    ]),
    ("Result: sequence length", [
        ("seq_dense", "First result: time as the text gets longer, on the C P U. Both axes are logarithmic, so the slope tells you how fast the cost grows. Dense attention has slope two. That's quadratic: double the length, four times the time."),
        ("seq_sparse", "Our local window and Longformer kernels have slope one. That's linear, because each query block always sees the same number of blocks, however long the text is."),
        ("seq_numbers", "At eight thousand tokens, dense takes seven hundred and sixty four milliseconds. Our local window takes fifty two. That's fourteen point seven times faster, almost exactly the fourteen and a half times less work. At sixteen thousand tokens, dense can't even run on this laptop. Ours takes one hundred and four milliseconds, thirteen times faster than PyTorch's own built in attention."),
    ]),
    ("Result: throughput and work", [
        ("work_chart", "Throughput tells the same story. Our local window handles about one hundred and fifty seven thousand tokens per second at every length, while dense drops to under eleven thousand. And the amount of computation: at sixteen thousand tokens, five hundred and fifty billion operations for dense, but only nineteen billion for ours. Twenty nine times less."),
    ]),
    ("Result: sparsity level", [
        ("sparsity_speedup", "Next, we fix the length at four thousand tokens and change how many blocks we keep. The dotted line is the ideal, one over density. All four patterns follow it. At twelve and a half percent, we're five to six times faster. At about three percent, around twelve times."),
        ("sparsity_masked", "And the pink line is masked dense, the lazy way. It never gets faster, at any density. It's actually a bit slower than plain dense, because hiding the unwanted part is extra work. This is the clearest result of the project: skipping helps, masking doesn't."),
    ]),
    ("Result: memory", [
        ("memory_chart", "Memory. Dense attention keeps two big tables, the scores and the weights. At eight thousand tokens our formula predicts four thousand and ninety six megabytes, and we measured four thousand and ninety nine. Our kernel only ever holds one small piece, so it needs about five megabytes at every length: roughly seven hundred and sixty times less. That's what lets sparse attention handle lengths that dense attention can't fit."),
    ]),
    ("Result: block size and threads", [
        ("blocksize_chart", "Block size is a trade off. Small blocks of sixteen follow a pattern closely, so ninety seven percent of the work is useful, but there are many tiny multiplies, and that is slow. Big blocks of one hundred and twenty eight waste nineteen percent of their work, but run faster, because computers prefer a few big operations to many small ones. We chose sixty four, as a balance."),
        ("threads_chart", "Threads. PyTorch uses four threads by default, one for each of the M4's four fast cores. That gives a two to two and a half times speed up over one thread. Adding the six slower efficiency cores gives nothing more."),
    ]),
    ("Result: the GPU", [
        ("gpu_chart", "The same code runs on the G P U by changing one setting. At eight thousand tokens, dense takes two hundred and ninety six milliseconds, ours twenty two: thirteen times faster. At sixteen thousand tokens, PyTorch's built in attention runs out of memory on the G P U, while ours finishes in thirty four milliseconds."),
        ("gpu_floor", "One limit we found: on the G P U, every small operation has a fixed start up cost. We launch a few small operations per query block, so below about twelve percent density, the time stops falling, at around ten milliseconds."),
    ]),
    ("Why not the ideal?", [
        ("why", "So why don't we reach the full ideal speed up? Four reasons. Each block has a fixed bookkeeping cost that doesn't shrink. Copying data costs time. Many small operations are less efficient than one big one, which is also why sparse is slower than dense when nothing is skipped. And uneven rows waste work if you force them into one shape. These are the computer architecture lessons of the project."),
    ]),
    ("Conclusion", [
        ("conclusion", "To sum up. We built a block sparse attention kernel and a benchmarking framework, and measured them on a laptop's C P U and G P U. With a sliding window, attention grows linearly instead of quadratically: fourteen point seven times faster than dense at eight thousand tokens, with about seven hundred and sixty times less memory. The saving only appears when skipped blocks are never computed. And how you organise the remaining work decides how close you get to the ideal."),
        ("end", "Thanks for watching."),
    ]),
]

if __name__ == "__main__":
    words = sum(len(t.split()) for _, b in CHAPTERS for _, t in b)
    print(f"{len(CHAPTERS)} chapters, {sum(len(b) for _, b in CHAPTERS)} beats, {words} words, ~{words/185:.1f} min")
