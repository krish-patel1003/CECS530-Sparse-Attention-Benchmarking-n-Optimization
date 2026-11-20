# Sparse Attention Benchmarking and Optimization

Block-sparse attention in **PyTorch** (CPU and Apple GPU), compared with dense attention by a
benchmarking framework. Start with **[STUDY_GUIDE.md](STUDY_GUIDE.md)**: every concept, every graph,
and likely questions, in plain words.

## Layout
```
sparse_attention/patterns.py    5 configurable sparsity patterns (dense, local, strided, longformer, random)
sparse_attention/attention.py   dense, masked dense, block-sparse v1 (gather) / v2 (views, final), padded batch
run_experiments.py              benchmarking framework: 7 experiments, --device cpu|mps
memory_probe.py                 peak memory of one call, in a fresh process
plot_results.py                 all graphs -> figures/
tests/test_correctness.py       168 correctness checks vs a float64 reference
results/ figures/ report/       outputs
```

## Run
```bash
pip install torch numpy pandas matplotlib pytest
python -m pytest -q tests
python run_experiments.py --list
python run_experiments.py --exp all                         # CPU, ~10 min
python run_experiments.py --exp seqlen sparsity ablation --device mps   # Apple GPU
python plot_results.py
```
Options: `--heads`, `--dim`, `--bs` (block size). Patterns and densities are set inside each
experiment and can be built directly with `patterns.make_pattern(name, N, BS, target_density=...)`.
