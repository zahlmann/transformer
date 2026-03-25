# EGGROLL Transformer — Agent Program

*You are an AI researcher. Your job: make EGGROLL (Evolution Strategies with low-rank
perturbations) training of a small transformer as FAST as possible while maintaining
quality. Focus on speed and memory optimization — kernel fusion, fp8, population
reduction, compilation tricks, anything that reduces wall-clock time. You work
autonomously — run experiments, log results, and keep going.*

**Your priority is SPEED OPTIMIZATION. Quality improvements are secondary —
only pursue them if a speed optimization opens a quality opportunity for free.**

---

## Current State (2026-03-25)

**Quality:** EGGROLL val_loss=2.44 (3-seed avg ~2.48) vs backprop+Adam 1.84.
Gap = 0.64 to backprop+Adam. Beats vanilla SGD backprop (2.44 vs 2.45).

**Speed: 175s for 10 epochs (was 444s).** 2.5x speedup achieved.
Backprop+Adam takes 4.1s. Speed gap is 43x (was 108x).

**Memory: 70MB (was 113MB).** Lower due to reduced population.

---

## Speed Budget Breakdown

Profiling shows 99% of time is in forward passes (the Triton kernel).
Current: 175s for 10 epochs = 17.5s/epoch. Each epoch has 61 batches.
Per batch: ~287ms. Per batch, kernel processes 8192 perturbation members (4096 pairs × 2 signs).

The Triton kernel runs in ~285ms per call (HALF_POP=4096, num_warps=4).
Gradient computation + Adam update: ~0.2ms (negligible after JIT).
QR decomposition: skipped (HALF_POP=4096 > vec_dim=2306, uses Gaussian).

Key metric: **time per batch** and **kernel time**.

---

## Files

```
program.md                    — this file (read first)
validate.py                   — LOCKED 3-seed validation (checks locked constants)
benchmark.py                  — fast single-seed comparison (EGGROLL vs backprop+Adam)
data.py                       — char-level Shakespeare dataset (65 vocab, 7842 train seqs)
model.py                      — decoder-only transformer (d=64, 1 layer, 2 heads, 66K params)
train_backprop.py             — backprop SGD baseline (LR=0.30, val_loss=2.45)
train_backprop_adam.py        — backprop Adam baseline (LR=3e-3, val_loss=1.84)
train_eggroll_triton.py       — BEST: fused Triton kernel EGGROLL (speed target)
train_eggroll_optimized.py    — JAX vmap EGGROLL (slower reference)
kernels/fused_transformer_ce.py — Triton kernel: full transformer forward + CE loss
kernels/cuda/                 — CUDA C++ kernel scaffolding (see below)
kernels/cuda/fused_transformer_ce.cu — CUDA kernel skeleton (TODO: implement body)
kernels/cuda/wrapper.py       — Python/JAX binding for CUDA kernel
kernels/cuda/test_kernel.py   — correctness test: CUDA vs Triton output
kernels/cuda/Makefile          — build: make -C kernels/cuda/
validate_kernel.py            — validates Triton kernel output vs JAX forward pass
benchmark_kernel.py           — benchmarks Triton vs JAX vmap speed
profile_triton.py             — time breakdown profiler (kernel vs gradient vs vec gen)
cuda_kernels_docs/            — Triton + jax-triton + CUDA C++ documentation
results.tsv                   — experiment log
```

Reference: `../eggroll_mnist/` has a similar project where a fused Triton kernel
achieved 2.16s (1.44x backprop) for MNIST. Study its `program.md`, `kernels/`,
and `mnist_eggroll_optimized.py` for optimization patterns.

---

## Fairness Rules (ENFORCED by validate.py)

LOCKED constants — do not change:
```python
D_MODEL = 64; N_HEADS = 2; N_LAYERS = 1; CONTEXT_LEN = 128
BATCH_SIZE = 128; EPOCHS = 10; TEMPERATURE = 2.0
```

Quality constraint: val_loss must stay ≤ 2.50 (3-seed avg). Any speed optimization
that degrades quality beyond this threshold is rejected.

---

## What Worked (Speed Session 2026-03-25)

1. **HALF_POP 8192→4096** (444s→220s, 2x): Halves kernel and gradient work.
   Quality maintained (val_loss 2.39 at seed=42). HALF_POP<4096 fails quality threshold.

2. **num_warps 8→4** (220s→180s, 18%): Fewer threads per block = more blocks per SM.
   Despite register spilling, better occupancy wins. num_warps=2 is much slower (423s),
   num_warps=16 also slower (370s).

3. **BLOCK_K 64→32** (180s→176s, 2%): Smaller FFN tiles = less register pressure per
   iteration. Marginal improvement.

## What Did NOT Work (Speed Session 2026-03-25)

- **FP8 tensor cores** (260s, slower): Register pressure from casts outweighs tensor
  core gains. Quality also degraded (2.53 vs 2.39).
- **Batch tiling** (BATCH_TILE=2 via tl.static_range): Triton compile timeout (>600s).
  The unrolled code was too large.
- **HALF_POP<4096**: All configs below 4096 fail the 2.50 quality threshold on 3-seed avg.
  Best was 3072+sigma=0.015 at 2.506 — barely over.
- **Adaptive HALF_POP** (4096 early, 2048/3072 late): Quality fails on seed 7 consistently.
  Best 3-seed avg was 2.507 — barely over.
- **CPU pre-shuffle**: No speed gain (GPU permutation is already fast).
- **Per-batch float() sync removal**: No speed gain (XLA handles pipelining through data deps).
- **VOCAB_PAD reduction** (128→80): Triton requires power-of-2 arange sizes. Can't reduce.

---

## What to Try Next (SPEED — reduce the 43x gap)

**The kernel is at ~47% of bf16 tensor core peak.** The gap is due to low occupancy
(register pressure from the 128×128 attention matrix). Focus on reducing register
pressure or changing the computation structure.

### HIGH PRIORITY: CUDA C++ kernel with FlashAttention tiling

**The CUDA C++ scaffolding is ready.** A skeleton kernel, Makefile, Python wrapper,
and test script are in `kernels/cuda/`. The Triton kernel cannot do FlashAttention
because Triton lacks shared memory control and can't slice register-resident 2D
tensors. CUDA C++ solves both problems.

**How to implement the CUDA kernel:**

1. **Start without FlashAttention** — port the Triton kernel logic to CUDA C++ line
   by line. Use `kernels/fused_transformer_ce.py` as reference. Get the test passing
   (`uv run kernels/cuda/test_kernel.py` compares CUDA vs Triton output).

2. **Add FlashAttention** — replace the full (128, 128) attention score computation:
   - Compute K, V for the full sequence → store to shared memory (32KB per head)
   - `__syncthreads()`
   - For each query tile (BLOCK_Q=32 query positions):
     * Compute Q_tile by having each thread compute Q for its assigned rows
     * For each KV tile (BLOCK_KV=32): load from shared mem, compute scores
       (32×32), apply causal mask, online softmax update
     * Finalize attention output for this tile
   - This reduces the attention tensor from (128,128)=16K to (32,32)=1K elements

3. **Wire it into JAX** — the wrapper.py has a placeholder. Complete the XLA
   custom_call lowering (see comments in wrapper.py). The `.so` is already built
   and `xla_client.register_custom_call_target` is ready.

4. **Benchmark** — replace the kernel call in `train_eggroll_triton.py` and run
   `uv run benchmark.py`.

**Build:** `make -C kernels/cuda/`
**Test:** `uv run kernels/cuda/test_kernel.py`
**Docs:** `cuda_kernels_docs/cuda_cpp/` has CUDA programming guide, tensor cores,
and JAX FFI reference.

---

### MEDIUM PRIORITY: Other approaches (if CUDA C++ is too complex)

1. **Tile attention over query positions in Triton**: Instead of computing full
   (128, 128) attention scores, compute (BLOCK_Q, 128) tiles. Use online softmax.

   **Challenge**: Q, K, V are computed on-the-fly from register-resident h_norm.
   Tiling requires either: (a) storing h_norm to shared memory and loading tiles,
   or (b) recomputing QKV per tile (wastes compute but reduces registers).

   The MNIST kernel doesn't have attention, so no reference implementation exists.
   Study FlashAttention papers (Dao et al. 2022) for the online softmax algorithm.

2. **Tile attention over key/value positions**: Alternative tiling axis. Process
   (128, BLOCK_KV) attention blocks with online softmax. Requires K, V to be
   available in tiles — need shared memory or recomputation.

### MEDIUM PRIORITY: Batch tiling (non-unrolled)

3. **Batch tiling with dynamic loop**: The `tl.static_range(2)` attempt caused Triton
   compile timeout because the unrolled code was too large. Try instead:
   - Use Triton's `tl.range()` (non-unrolled loop) if available
   - Or write a C++/CUDA kernel wrapper that handles the batch loop
   - Or use Pallas (JAX native Triton alternative) which may have cheaper JIT

### LOWER PRIORITY

4. **Split kernel into attention + FFN**: Separate the fused kernel into two smaller
   kernels. Trade HBM traffic for better occupancy. The intermediate (h after attention)
   is 128×64×4 = 32KB per perturbation member. With 4096 members × 2 signs × 32KB =
   256MB through HBM. At 700 GB/s, that's ~0.4ms per batch — potentially worth it
   if the occupancy improvement is significant.

5. **Population reduction below 4096**: All attempts failed quality threshold (3-seed
   avg ≤ 2.50). Best was HALF_POP=3072+sigma=0.015 at 2.506. Could revisit with:
   - Different optimizer (e.g., AdaGrad, RMSProp)
   - Gradient accumulation across batches
   - Better fitness shaping (rank-based, top-k selection)

6. **All-in-one JIT via lax.scan**: Wrap all epochs and batches in a single JIT.
   Python loop overhead is only ~6ms total (negligible). Main benefit would be if
   XLA can optimize kernel scheduling or overlap operations.

### ALREADY ELIMINATED

These have been tried and don't work — don't repeat:
- FP8 tensor cores (slower + quality loss)
- num_warps=2 (much slower), num_warps=8 (baseline), num_warps=16 (slower)
- BLOCK_K=64 (baseline), BLOCK_K=32 (current, marginal improvement)
- VOCAB_PAD<128 (Triton requires power-of-2 arange)
- CPU pre-shuffle (no speed gain)
- Per-batch float() sync removal (no speed gain)
- Adaptive HALF_POP (quality fails on seed 7)

---

## What Already Works (do not change unless improving speed)

- bf16 forward passes with fp32 perturbation math
- Constant label smoothing alpha=0.50
- Adam optimizer (β1=0.9, β2=0.999, eps=1e-6) with bias correction
- Sigma=0.02, LR=0.010, no LR decay
- Fused Triton kernel (Grid: HALF_POP × BATCH × 2, num_warps=4, BLOCK_K=32)
- Per-subgroup Winsorized z-score (K=8, clip ±2.0)
- Rank-1 perturbation compression (vec_dim=2306, 28.8x compression)

---

## Current Best Hyperparameters

```python
HALF_POP = 4096          # antithetic pairs → pop=8192 (was 8192→16384)
SIGMA_START = 0.02       # perturbation scale
SIGMA_DECAY = 0.998      # per epoch
LR_START = 0.010         # Adam learning rate
LR_DECAY = 1.0           # no decay (Adam self-adjusts)
ALPHA = 0.50             # label smoothing (CONSTANT)
TEMPERATURE = 2.0        # CE temperature (LOCKED)
MOMENTUM = 0.9           # Adam β1
ADAM_BETA2 = 0.999       # Adam β2
ADAM_EPS = 1e-6           # Adam epsilon
N_SUBGROUPS = 8          # Winsorized z-score groups
CLIP_RANGE = 2.0         # z-score clipping
```

---

## Architecture Details

```
Model: decoder-only transformer
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128, vocab_size: 65 (character-level Shakespeare)
Parameters: 66,368
Perturbation vec_dim: 2306 (28.8x compression via rank-1)
```

---

## Results History (10 epochs)

### Backprop baselines

| Optimizer | LR | val_loss | ppl | Time | Memory |
|-----------|-----|----------|-----|------|--------|
| SGD | 3e-1 | 2.45 | 11.6 | 1.3s | 300MB |
| **Adam** | **3e-3** | **1.84** | **6.3** | **4.1s** | **160MB** |

### EGGROLL results

| Config | val_loss | ppl | Time | Notes |
|--------|----------|-----|------|-------|
| JAX vmap, pop=4096 | 2.67 | 14.4 | 540s | before Triton kernel |
| Triton, pop=2048 | 2.67 | 14.5 | 119s | 4.5x kernel speedup |
| Triton, pop=8192, SGD+mom | 2.64 | 14.0 | 220s | |
| Triton, pop=8192, Adam, σ=0.02 | 2.50 | 12.2 | 220s | Adam breakthrough |
| Triton, pop=16384, Adam, σ=0.02 | 2.37 | 10.7 | 444s | previous best |
| **Triton, pop=8192, Adam, σ=0.02, nw=4, BK=32** | **2.44** | **11.4** | **175s** | **current best** |

---

## Setup

1. `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read this file, then `../eggroll_mnist/program.md` for speed optimization patterns
3. `uv run benchmark.py` to get current baseline numbers
4. `uv run profile_eggroll.py` to understand the time breakdown
5. Begin the experiment loop

---

## Experiment Loop

```
1. Pick a speed optimization from "What to Try Next"
2. Implement it
3. git add -A && git commit -m "description"
4. uv run benchmark.py
5. Check: did time decrease? Is val_loss still ≤ 2.50?
6. If crashed: fix and retry
7. If faster AND quality maintained: git push, update this file
8. If slower or quality degraded: git reset --hard HEAD~1
9. Go to step 1
```

Never stop to ask. The cost of a failed run is ~8 minutes. Keep experimenting.

---

## Logging (results.tsv)

Tab-separated, one row per run, append-only.

Columns: `timestamp commit seed val_loss perplexity training_time_s peak_memory_mb status description`
