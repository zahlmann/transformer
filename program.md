# EGGROLL Transformer — Agent Program

*You are an AI researcher working on EGGROLL — Evolution Strategies with low-rank
perturbations for training transformers WITHOUT backpropagation. Before diving into code,
read the "Where This Could Lead" section below to understand the bigger picture.*

## Where This Could Lead (Problem Space — Read First)

We've spent two intensive sessions pushing EGGROLL quality on a small transformer. The
quality gap to backprop is real (val_loss 2.38 vs 1.84) and appears fundamental for this
setup. Before trying more solution-space optimizations, step back and ask: **what is this
project actually good for, and where should it go?**

### What We've Actually Built

1. **A fused Triton kernel that runs an entire transformer forward pass in one kernel call.**
   Embedding → LayerNorm → Multi-Head Attention → FFN → Output → CE Loss, all fused. No
   HBM round-trips between layers. This is valuable independent of ES training — it's a
   template for fused inference kernels, forward-mode AD kernels, and custom training loops.

2. **A rank-1 perturbation framework** that compresses 66K-parameter perturbations into
   2306-dimensional vectors (28.8× compression). The kernel applies rank-1 weight updates
   via efficient matvec + outer product operations, avoiding full matrix perturbation.

3. **A dual-number Triton kernel** (`kernels/fused_transformer_ce_dual.py`) that computes
   exact directional derivatives via forward-mode AD, fused into the same single-kernel
   architecture. Validated against jax.jvp (1.00±0.04 accuracy). First known implementation
   of fused forward-mode AD in a Triton transformer kernel.

### Where Backprop-Free Training Actually Matters

EGGROLL can't match backprop at equal compute on standard transformers. That's not a bug —
it's a statement about the problem. The right question isn't "how to make ES match backprop"
but **"where does avoiding backprop give you something backprop can't?"**:

- **Non-differentiable objectives.** RLHF reward models, discrete sampling (hard attention,
  VQ-VAE codebook selection), binary masks, integer programs. ES optimizes the ACTUAL
  objective, not a differentiable surrogate. The fused kernel can evaluate any forward-pass
  objective, not just CE loss.

- **Communication-free distributed training.** Each GPU independently evaluates perturbation
  members. No gradient synchronization, no all-reduce, no pipeline bubbles. On 1000+ GPUs,
  backprop's communication overhead dominates; ES scales linearly. The kernel's massive
  parallelism (14K+ concurrent evaluations) maps naturally to multi-GPU.

- **Memory-constrained deployment.** No activation storage for backward pass. EGGROLL uses
  70-103MB vs backprop's 160MB+ on this model. For billion-parameter models on edge devices,
  this gap widens dramatically. ES needs only the forward pass, which can be heavily optimized
  (quantized, pruned, fused).

- **Hardware without backward-pass support.** Custom accelerators, FPGAs, neuromorphic chips,
  analog compute. If you can run a forward pass, you can run EGGROLL. The fused kernel shows
  how to structure the computation for any hardware target.

- **Model editing and patching.** Rank-1 perturbations are structurally identical to LoRA
  adapters (low-rank weight updates). EGGROLL could be used for gradient-free LoRA fine-tuning
  — finding good low-rank weight edits without backprop. This is relevant for on-device
  personalization, model merging, and task adaptation.

### Concrete Next Directions (pick one)

**A. Scale to a real model (most impactful).** Take the fused kernel architecture and apply
it to a GPT-2-small (117M params) or similar. The rank-1 perturbation framework scales
well (vec_dim grows as sqrt(params), not linearly). The key question: does the quality gap
shrink or grow with model size? Theory suggests it shrinks (larger models have smoother
loss landscapes), but this hasn't been tested.

**B. Non-differentiable fine-tuning demo.** Add a non-differentiable objective (e.g., BLEU
score for text generation, exact-match for QA) and show EGGROLL optimizing it directly
while backprop can't. This is the strongest "why ES matters" demonstration.

**C. Multi-GPU scaling experiment.** Run EGGROLL on 4-8 GPUs with zero communication.
Compare wall-clock time and quality against data-parallel backprop with gradient all-reduce.
The crossover point (where ES becomes faster than backprop) is a publishable result.

**D. Gradient-free LoRA.** Adapt the rank-1 perturbation framework for LoRA-style
fine-tuning of a pre-trained model. No backprop needed — just forward passes through the
frozen model with rank-1 weight deltas. Compare with standard LoRA (which requires backprop).

**E. Fused inference kernel (pivot away from training).** The fused Triton kernel is
independently valuable for inference. Strip out the ES machinery and benchmark as a
single-kernel inference engine vs standard multi-kernel PyTorch/JAX inference. Could be
packaged as a standalone tool.

### What NOT to Do

- Don't keep tuning hyperparameters on this small model. 30+ experiments have thoroughly
  explored the HP space. The ceiling is ~2.38 val_loss (3-seed avg) at ~300s budget.
- Don't try to close the gap to backprop (1.84) on this architecture. The gap is fundamental
  to rank-1 ES with 10 epochs on a 66K-param model. The math says so, and the experiments
  confirm it.
- Don't add complexity (CMA-ES, NES, etc.) without a clear hypothesis for why it would help
  given the rank-1 constraint and 610-update budget.

---

**MANDATORY CLEANUP RULES (before every merge to main):**
1. **One train_eggroll.py** — the best EGGROLL implementation. No duplicates.
2. **One train_backprop.py** — the Adam baseline. No SGD variant, no sweeps.
3. **Delete failed experiments** — HP sweeps, test scripts for approaches that
   didn't work, intermediate/scratch files. Document what failed in this file
   under "ALREADY ELIMINATED", then delete the code.
4. **No sweep/search scripts on main** — run them on branches, record results
   in this file, delete the scripts before merging.
5. **Kernel code** — the fused Triton kernel in `kernels/` is production. The CUDA
   kernel in `kernels/cuda/` is working infrastructure (correct, JAX binding works,
   needs speed optimization). Keep both. Delete truly broken experiments only.
6. **Keep**: data.py, model.py, benchmark.py, validate.py, profile_triton.py,
   validate_kernel.py, results.tsv, program.md, README.md

---

## Current State (2026-03-29)

**Quality:** EGGROLL val_loss=2.38 (3-seed avg 2.376) vs backprop+Adam 1.84.
Gap = 0.54 to backprop+Adam (was 0.65). Improved from 2.49 via HALF_POP=7168.

**Speed: 272s for 10 epochs (was 153s at HALF_POP=4096).** Slower due to larger population,
but within the 300s quality budget. Backprop+Adam takes 4.1s.

**Memory: 103MB (was 70MB).** Higher due to larger population.

---

## Speed Budget Breakdown

Profiling shows 99% of time is in forward passes (the Triton kernel).
Current: 156s for 10 epochs = 15.6s/epoch. Each epoch has 61 batches.
Per batch: ~243ms. Per batch, kernel processes 8192 perturbation members (4096 pairs × 2 signs).

The Triton kernel runs in ~243ms per call (HALF_POP=4096, num_warps=4, FFN dynamic loop).
Gradient computation + Adam update: ~0.2ms (negligible after JIT).
QR decomposition: skipped (HALF_POP=4096 > vec_dim=2306, uses Gaussian).

Key metric: **time per batch** and **kernel time**.

---

## Files

```
program.md                      — this file (read first)
README.md                       — project overview
train_eggroll.py                — EGGROLL training (speed target)
train_backprop.py               — backprop+Adam baseline (LR=3e-3, val_loss=1.84)
benchmark.py                    — single-seed comparison (EGGROLL vs backprop)
validate.py                     — 3-seed quality validation (checks locked constants)
data.py                         — char-level Shakespeare dataset (65 vocab, 7842 train seqs)
model.py                        — decoder-only transformer (d=64, 1 layer, 2 heads, 66K params)
kernels/fused_transformer_ce.py — fused Triton kernel: full forward + CE loss
profile_triton.py               — kernel time breakdown profiler
validate_kernel.py              — validates Triton kernel output vs JAX forward pass
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

Quality target: close the gap from val_loss 2.49 toward backprop's 1.84. The current
floor for speed-only optimization was val_loss ≤ 2.50 — now we want to push WELL below
that. Speed may regress moderately (up to ~200s) if quality improves significantly.

---

## What Worked (Speed Session 2026-03-25)

1. **HALF_POP 8192→4096** (444s→220s, 2x): Halves kernel and gradient work.
   Quality maintained (val_loss 2.39 at seed=42). HALF_POP<4096 fails quality threshold.

2. **num_warps 8→4** (220s→180s, 18%): Fewer threads per block = more blocks per SM.
   Despite register spilling, better occupancy wins. num_warps=2 is much slower (423s),
   num_warps=16 also slower (370s).

3. **BLOCK_K 64→32** (180s→176s, 2%): Smaller FFN tiles = less register pressure per
   iteration. Marginal improvement.

4. **FFN dynamic loop tl.range** (175s→156s, 10.6%): Changed FFN K-tiling loop from
   `tl.static_range` to `tl.range`. Prevents compiler from unrolling 8 iterations,
   reducing register pressure from 255 regs + 340 bytes spill to 255 regs + 0 spill.
   Quality maintained (3-seed avg 2.48). Head loop dynamic range helps speed marginally
   (→153s) but fails 3-seed quality threshold by 0.004-0.008.

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

## What Did NOT Work (Speed Session 2026-03-25 part 2)

- **Triton FlashAttention via tl.gather** (308ms vs 285ms baseline, SLOWER): Triton 3.6
  supports tl.ravel + tl.gather + tl.reshape to extract KV tile rows from register-resident
  tensors. BLOCK_KV=64 gave 278ms isolated kernel benchmark but 181s full training (worse
  than 176s baseline). The gather operation compiles to cross-thread register shuffles that
  add more overhead than the register pressure reduction saves.
- **Tiled output projection** (287ms vs 285ms, no improvement): Tiling the (128,128) logits
  matrix with online log-sum-exp for CE loss. Loop overhead negates register savings.
- **FFN pipelining** (tl.range num_stages=2): 545ms, much slower. Dynamic loop with
  pipelining doesn't work for the FFN body structure.
- **HALF_POP=3072 HP search** (val_loss 2.56 3-seed avg, FAIL): Searched sigma
  [0.012-0.025], lr [0.008-0.015], alpha [0.4-0.6], sigma_decay [0.995-1.0], LR schedules
  (cosine, warmup+cosine). Best single-seed was 2.4946 (sigma=0.018) but 3-seed avg is
  2.5638. Population reduction not viable without algorithmic changes.
- **CUDA kernel (scalar FP32)** (3860ms vs 271ms Triton, 14x SLOWER): Full CUDA kernel
  with FlashAttention via shared memory. Correctness validated (max diff 0.0004). But
  without WMMA tensor cores, scalar FP32 can't compete with Triton's bf16 tensor cores.

## What Did NOT Work (Speed Session 2026-03-25 part 3)

- **fp16/bf16 output projection accumulator** (175.8s, no improvement): out_dtype in tl.dot
  doesn't meaningfully reduce register pressure since the result is immediately cast to f32.
  bf16 not supported as out_dtype; fp16 works but no speed gain.
- **Dynamic head loop tl.range** (153s but 3-seed avg 2.504-2.508, QUALITY FAIL by 0.004-0.008):
  Marginal speed gain but just barely fails quality. HP tuning (sigma=0.022/0.025, lr=0.012)
  could not compensate.
- **HALF_POP=3072 with adaptive alpha schedule** (133-134s, val_loss 2.69-2.84, FAIL):
  MNIST-inspired adaptive alpha (decay 0.5/epoch and 0.9/epoch) doesn't transfer to transformer.
  At HALF_POP=3072, quality fails regardless of alpha schedule or HP combination.
- **num_warps=8 with FFN dynamic loop** (197.1s, much slower): Lower per-thread register
  count from more threads doesn't help — thread overhead dominates.
- **BLOCK_K=16 with FFN dynamic loop** (171.9s, slower than BK=32 at 156s): Smaller tiles
  reduce tensor core efficiency and increase loop iterations.
- **BLOCK_K=64 with FFN dynamic loop** (168.7s, slower): Larger tiles increase per-iteration
  register pressure despite fewer iterations.
- **maxnreg=248/232** (256.1ms/252.6ms vs baseline 243ms, SLOWER): Properly monkey-patched
  jax_triton to pass maxnreg to Triton compiler. Compiles and runs but spill overhead
  outweighs any occupancy benefit. Kernel is too compute-bound for spill-induced occupancy.
- **Fewer batches per epoch** (48 batches: 123s but 2.56 val_loss; 56 batches: 144s but 2.51):
  Reducing gradient updates degrades quality faster than it saves time.

## What Worked (Quality Session 2026-03-26)

7. **HALF_POP=7168, sigma=0.020** (2.49→2.38, 3-seed avg 2.490→2.376): Larger population
   directly improves gradient quality. Optimal sigma shifted from 0.022 to 0.020 at this
   population size. Time increased from 153s to 272s (within 300s quality budget).

## What Worked (Speed Session 2026-03-26)

5. **Dynamic head loop tl.range + sigma=0.022** (156s→153s, 2.0%): Previous attempts
   with dynamic head loop failed quality by 0.004-0.008 at sigma=0.02. Increasing sigma
   from 0.02 to 0.022 compensated, bringing 3-seed avg from 2.504 to 2.490 (passes ≤2.50).
   Dynamic range prevents head loop unrolling, reducing register-related overhead.

6. **Per-batch float() sync removal** (built into the 153s): Replaced `eloss += float(pl)`
   with `eloss = eloss + pl` to avoid GPU-host sync on every batch. Saves ~1.5s.

## What Did NOT Work (Speed Session 2026-03-26)

- **Sequence-tiled kernel (SEQ_TILE=64)**: Split 128 positions into two 64-position tiles.
  Tile 0 achieved 240 regs/thread (2 blocks/SM!) and 139ms. Tile 1 with FlashAttention
  recompute stuck at 255 regs/thread (1 block/SM), 151ms. Total 290ms > original 243ms.
  ROOT CAUSE: two-kernel launches add overhead, and smaller matmuls (64×64 vs 128×128)
  have lower arithmetic intensity, negating the occupancy benefit. Even tile 0 at 2 blocks/SM
  is SLOWER per-position than the original at 1 block/SM.
- **Restructured tile 1 liveness** (compute K_lower/V_lower → stage 1 → K_upper/V_upper → stage 2):
  Still 255 regs — compiler can't reduce below max even with optimal variable ordering.
- **Dynamic head loop for tile 1** (tl.range vs tl.static_range): Reduced tile 1 from 228ms
  to 151ms (no unrolling overhead) but registers stayed at 255.
- **Local attention for tile 1** (no cross-tile attention): val_loss=2.89, quality destroyed.
  Also 178s total — slower due to two-kernel overhead.
- **jax.lax.scan epoch loop** (compile entire epoch as single XLA graph): 154.0s, no improvement.
  Per-batch dispatch overhead is ~0.1ms, negligible vs 250ms kernel time. MNIST benefited
  because per-batch time was ~1ms. Doesn't transfer to expensive kernel.
- **Per-batch float() sync removal** (156s → 154s, 1.3% speedup): Marginal improvement from
  removing GPU-host synchronization on every batch. Accumulated proxy loss as JAX array
  instead of converting to Python float.

## What IS Ready (for next session)

- **CUDA kernel infrastructure**: Full kernel in `kernels/cuda/fused_transformer_ce.cu`
  (664 lines), JAX XLA custom_call binding in `kernels/cuda/wrapper.py`. Builds, runs,
  produces correct output. Only missing WMMA tensor cores for the big matmuls.
  - Build: `make -C kernels/cuda/`
  - Test: `uv run python -c "..."` (see test_kernel.py)
  - Uses: PyCapsule for custom_call registration, scalars packed in opaque struct
  - Key fix: scalars (sigma, alpha, temperature) must be in opaque struct, NOT as
    separate GPU buffer operands (XLA 0.9.2 can't handle scalar GPU buffers)

---

## What to Try Next (QUALITY — close the gap from 2.49 to 1.84)

**The speed is at Triton's limit (153s, 255 regs, 47% utilization).** Further kernel speedups
require CUDA PTX — not worth pursuing until quality is addressed. The priority is now:
close the val_loss gap from 2.49 (EGGROLL) to 1.84 (backprop+Adam).

**Why EGGROLL lags backprop by 0.65 val_loss:**
1. ES gradient estimates are inherently noisier than exact backprop gradients
2. With only 10 epochs × 61 batches = 610 gradient updates, each update must be high quality
3. The rank-1 perturbation restricts gradient directions (2306-dim vs 66K-dim param space)
4. Label smoothing α=0.50 was tuned for speed (stable gradients), not quality (sharp learning)

**Concrete approaches to try (ordered by expected impact):**

## What to Try Next — Quality Improvement (for next agent)

### Understanding the Quality Gap

EGGROLL val_loss=2.49, backprop val_loss=1.84. That's a 0.65 gap = 35% higher loss.

Generated text comparison (after 10 epochs, prompt "ROMEO:\nO, "):
- **EGGROLL**: repetitive, blobby ("the the ther forer, wherou hathe he the beporeant")
- **Backprop**: more structure, learns newlines/punctuation/formatting ("I pay herow have Are they their me.\nAN Rome")

The gap comes from: ES gradients are ~100x noisier than backprop with only 610 updates.

### Approach 1 — Learning Rate Schedule (HIGHEST EXPECTED IMPACT)

Currently LR=0.010, constant. Backprop uses LR=3e-3 with Adam. Try:
- **Linear warmup + cosine decay**: warm up from 0 to LR_MAX over first 2 epochs, then cosine
  decay to LR_MIN. Warmup helps when early gradients are noisy (ES is inherently noisy).
- **Higher peak LR**: With warmup, can try LR_MAX=0.015-0.030 without divergence.
- **Cosine annealing**: LR starts high, decays smoothly. Late epochs do fine-tuning.

Implementation: change `lrs_sched` computation in train_eggroll.py. No kernel changes needed.

### Approach 2 — Guided ES / Gradient Memory

Use the gradient from the PREVIOUS step to bias perturbation directions toward the
descent direction. This makes each perturbation more informative.

**Idea**: Instead of pure random perturbations, use a mix of:
- 90% random directions (exploration)
- 10% of vectors aligned with the previous gradient (exploitation)

This requires storing the previous gradient and mixing it into the perturbation matrix.
Minimal speed impact (gradient is already computed, just add a linear combination).

### Approach 3 — Orthogonal Perturbations via Structured Random Matrices

Currently HALF_POP=4096 > vec_dim=2306, so we use Gaussian perturbations. But random
Gaussian vectors are NOT orthogonal — they have random overlap, wasting perturbation budget.

**Options:**
- Use QR decomposition (already implemented but skipped when HALF_POP > vec_dim). What if
  we generate 2306 orthogonal vectors + 1790 random ones? This gives full coverage of the
  parameter space + extra exploration.
- Use Hadamard-based random rotations (O(n log n) instead of O(n²) for QR).
- Use block-orthogonal: split 4096 vectors into 2 blocks of 2048, each block is
  independently orthogonalized. Less compute than full QR, better coverage than Gaussian.

### Approach 4 — Sigma Schedule Tuning

Currently sigma decays as σ_start × 0.998^epoch (barely decays: 0.022 → 0.0216 over 10 epochs).
More aggressive sigma decay could help: start large (explore) and shrink (exploit).

- **σ_start=0.04, σ_decay=0.85**: 0.040, 0.034, 0.029, 0.025, 0.021, 0.018, 0.015, 0.013, 0.011, 0.009
- **σ_start=0.03, σ_end=0.01, cosine**: smooth transition from exploration to exploitation

Higher initial sigma explores more of the loss landscape; lower final sigma gives
sharper gradient estimates for fine-tuning.

### Approach 5 — Population-Weighted Gradient Estimation

Currently all perturbation members contribute equally to the gradient (after z-scoring).
What if we weight them by fitness? Better-performing perturbations get more influence.

- **Rank-based weighting**: top-25% perturbations get 4x weight, bottom-25% get 0.25x
- **Fitness-proportional**: weight proportional to exp(-fitness * temperature)
- This is essentially moving toward CMA-ES (covariance matrix adaptation)

### Approach 6 — Multi-Point Evaluation per Perturbation

Currently each perturbation is evaluated at +σ and -σ (2 points). What about evaluating
at +σ, -σ, +2σ, -2σ (4 points)? This gives a better gradient estimate per perturbation
at the cost of 2x more forward passes.

With 4-point evaluation: kernel grid is (4096, 128, 4) instead of (4096, 128, 2).
Time roughly doubles (from 153s to ~300s) but gradient quality improves significantly.
Trade-off: 2x slower but maybe 1.5x better convergence per update.

**Alternative**: use the extra evaluations ONLY for the top-K perturbations that showed
the most promise in the 2-point evaluation. This is "progressive refinement."

### Approach 7 — Weight Initialization

The model starts from random initialization. Better init might give EGGROLL a head start:
- **Kaiming/He init** scaled for the attention architecture
- **Fixup init**: scale residual connections to prevent gradient explosion
- **Smaller init scale**: ES works better when the loss landscape is smoother (smaller weights)

### Approach 8 — Hybrid Training — REJECTED

**DO NOT USE hybrid backprop warmup.** User explicitly rejected this approach. The goal is
pure EGGROLL (evolution strategies) quality, not backprop-assisted training.

Tested results (for reference): 5 BP warmup + 5 EGGROLL at HALF_POP=8192 achieved
3-seed avg 2.204 (vs pure EGGROLL 2.49). But the improvement came from backprop, not
from EGGROLL improving — EGGROLL actually degraded the BP-trained model before partially
recovering. This approach is off-limits.

### Approach 9 — Adam Hyperparameter Tuning

Current Adam: β1=0.9, β2=0.999, ε=1e-6, lr=0.010.
These were tuned for speed optimization. For quality:
- **Higher β2** (0.9999): smoother second moment, less aggressive adaptation
- **Lower ε** (1e-8): default Adam, might help precision
- **Higher LR** with warmup: faster learning when gradients are informative
- **Weight decay** (AdamW): regularization might help generalization

### Profiling Commands
```bash
uv run benchmark.py              # speed + quality check
uv run validate.py               # 3-seed quality validation
uv run train_eggroll.py --seed 42   # single-seed quick test
uv run profile_triton.py         # kernel time breakdown
```

### Speed Budget for Quality Work
Current: 153s. The kernel runs at 240ms/batch and is near its limit.
- Acceptable: up to ~200s (e.g., from higher HALF_POP or more compute per batch)
- Maximum: ~300s (e.g., from 4-point evaluation or other 2x changes)
- No-go: anything > 300s is too slow for the experiment loop

### ALREADY ELIMINATED (don't repeat)

These have ALL been tried and don't work:
- FP8 tensor cores (slower + quality loss)
- num_warps=2 (much slower), num_warps=8 (baseline), num_warps=16 (slower)
- BLOCK_K=64 (baseline), BLOCK_K=32 (current, marginal improvement)
- VOCAB_PAD<128 (Triton requires power-of-2 arange)
- CPU pre-shuffle (no speed gain)
- Per-batch float() sync removal (no speed gain)
- Adaptive HALF_POP (quality fails on seed 7)
- Triton FlashAttention via tl.gather/tl.ravel/tl.reshape (8% slower, gather overhead)
- Tiled output projection with online log-sum-exp CE (no improvement)
- FFN pipelining via tl.range num_stages=2 (2x slower)
- Split kernel into attn + ffn_ce (12% slower, HBM scratch traffic)
- HALF_POP=3072/3584 with HP search (quality fails 3-seed threshold)
- LR schedules (cosine, warmup+cosine) at HALF_POP=3072 (no quality improvement)
- Rank-based fitness shaping (gradient scaling issues, diverged)
- maxnreg via monkey-patched jax_triton (doesn't affect compilation due to caching)
- maxnreg=232/248 via proper monkey-patch (compiles but spill overhead > occupancy gain: 253-256ms vs 243ms baseline)
- bf16 attention weights (no speed improvement)
- fp16 output projection accumulator (out_dtype=tl.float16, no speed improvement)
- fp16 attention score accumulator (out_dtype=tl.float16, no speed improvement alone)
- num_stages=1/2/3 (no difference, kernel is compute-bound)
- CUDA scalar kernel (14x slower without tensor cores)
- CUDA WMMA kernel (11x slower, shared memory staging overhead + scalar non-matmul code)
- Dynamic head loop tl.range (153s speed but 3-seed avg 2.504-2.508, fails quality by 0.004-0.008)
- Dynamic head loop + fp16 attn + sigma/LR tuning (can't compensate for quality loss)
- Adaptive alpha schedule (0.5/epoch decay) at HALF_POP=3072 (val_loss 2.84, total fail)
- Gentle alpha schedule (0.9/epoch decay) at HALF_POP=3072 (val_loss 2.69, still fails)
- BLOCK_K=16 with dynamic FFN loop (171.9s, slower than BLOCK_K=32 at 156s)
- BLOCK_K=64 with dynamic FFN loop (168.7s, slower, larger tiles increase register pressure)
- num_warps=8 with dynamic FFN loop (197.1s, much slower)
- Fewer batches per epoch (48: quality fails; 56: borderline fail at 2.51 3-seed)
- Sequence-tiled kernel SEQ_TILE=64 (tile 0: 240 regs/139ms, tile 1: 255 regs/151ms, total 290ms > 243ms baseline)
- Sequence-tiled with local attention (val_loss=2.89, quality destroyed, also slower at 178s)
- FlashAttention recompute for tile 1 with restructured liveness (still 255 regs)
- Dynamic head loop for tile 1 only (151ms but still 255 regs)
- jax.lax.scan epoch loop (154s, no improvement — dispatch overhead negligible vs kernel time)
- HALF_POP=3840/3968/4032 with dynamic head + sigma 0.022-0.028 (3-seed avg 2.51-2.53, FAIL)
- Adaptive label smoothing α_decay=0.70 (val_loss 2.72, quality destroyed)
- Adaptive label smoothing α_decay=0.95 (val_loss 2.47 at seed=42, worse than constant α=0.50)
- Constant alpha=0.20 (2.56), alpha=0.25 (2.53), alpha=0.35 (2.52) — all worse than α=0.50 baseline
- Constant alpha=0.30 (2.49) — matches baseline, no improvement
- LR=0.015 constant (2.64 with oscillation), LR=0.005 constant (2.55 still declining) — 0.010 is optimal
- Warmup+cosine LR schedule (LR_MAX=0.020, warmup 2 epochs): 2.59, warmup wastes early epochs
- Cosine LR decay (0.020→0.002): 2.59, high initial LR hurts ES convergence
- Aggressive sigma schedule (0.04→0.85^epoch): 2.66, early epochs too noisy
- Cosine sigma schedule (0.03→0.01): 2.57, start too noisy
- Hybrid QR orthogonal perturbations at HALF_POP=4096 (2306 ortho + 1790 random): 2.50, no improvement
- Pure QR HALF_POP=2304: 2.57 at 94s, faster but worse quality
- Guided ES (bias perturbations toward prev gradient, GUIDE_SCALE=0.3): 2.78, catastrophic
- Momentum β1=0.95: 2.51, too much smoothing for 10 epochs
- AdamW weight decay=0.01 with α=0.30: 2.51, no improvement
- Smaller init scale 0.5x: 2.60, model learns slower from flat region
- Hybrid backprop warmup: REJECTED by user — pure EGGROLL only
- At HALF_POP=7168: sigma=0.018 (2.48), sigma=0.022 (2.35), sigma=0.025 (2.46) — 0.020 is optimal
- At HALF_POP=7168: LR=0.008 (2.44), LR=0.012 (2.40) — 0.010 still optimal
- At HALF_POP=7168: N_SUBGROUPS=4 (2.32), N_SUBGROUPS=16 (2.32) — insensitive
- EMA parameter averaging (decay=0.99, 0.999): both worse — trajectory still descending
- SWA last-3-epochs averaging: worse (2.34 vs 2.31) — trajectory still descending
- Adam beta2=0.9999: same as 0.999, no impact
- Adam eps=1e-8: slightly worse than 1e-6
- HALF_POP=3584 with 2x data passes: 2.42, worse than 7168×1 (gradient quality > quantity)
- HALF_POP=8192 at sigma=0.020: 2.41 at 314s (over budget and seed-noisy)
- Forward-mode AD via jax.jvp + lax.scan: exact directional derivatives but 12x slower per
  direction than Triton kernel (sequential processing vs massive parallelism). N_DIRS=256 gives
  val_loss=2.73 at 118s — much worse than kernel's HALF_POP=7168 at 2.31/272s. The kernel's
  parallelism is the key advantage; forward-mode AD can't match it without a dual-number kernel.
- Forward-mode AD with full-rank perturbations (no rank-1 compression): val_loss=2.75 at 118s.
  WORSE than rank-1 — higher dimensionality (66K vs 2306) means sparser gradient coverage per
  direction. Rank-1 compression is a feature, not a bug, for ES sample efficiency.
- Forward-mode AD with chunked vmap (64/chunk): 3.6GB memory, 151s. OOM at full vmap (256).
- **Dual-number Triton kernel** (fused forward-mode AD): tangent accuracy 1.00±0.04 vs jax.jvp.
  At N_DIRS=7168: val_loss=2.42 at 354s (vs finite-diff 2.31 at 272s). 0.11 worse + 30% slower.
  Root cause: accumulated bf16 precision loss through ~10 tangent matmuls. Finite differences
  benefit from antithetic pairing (+σ/-σ) which cancels common-mode bf16 rounding errors.
  The dual kernel is a technically correct implementation but loses to finite differences
  at bf16 precision on this small model. Code in kernels/fused_transformer_ce_dual.py.
- Population scheduling (4096 early + 10240 late, or reverse): same total compute as uniform
  7168, but worse quality (2.36 and 2.32 vs 2.31). Uniform allocation is optimal.
- Alpha=0.10 constant at HALF_POP=7168: 2.59, much worse — transformer needs high smoothing
- Adaptive alpha 0.50 → 0.10 (decay=0.85/epoch): 2.49, worse — low alpha causes overfitting
- Sigma decay 0.025→0.011 (decay=0.92/epoch) at HALF_POP=7168: 2.49, worse than constant 0.020
- LR warmup (0.006→0.010 over 2 epochs) at HALF_POP=7168: 2.44, worse — wastes early epochs
- SGD with Nesterov momentum (LR=0.10): diverged completely
- Raw fitness diffs (no z-scoring): 2.45, z-scoring is essential
- CLIP_RANGE=1.5: 2.35, CLIP_RANGE=2.5: 2.41, CLIP_RANGE=3.0: 2.43 — 2.0 is optimal

---

## What Already Works (do not change unless improving speed)

- bf16 forward passes with fp32 perturbation math
- Constant label smoothing alpha=0.50
- Adam optimizer (β1=0.9, β2=0.999, eps=1e-6) with bias correction
- Sigma=0.02, LR=0.010, no LR decay
- Fused Triton kernel (Grid: HALF_POP × BATCH × 2, num_warps=4, BLOCK_K=32, FFN+head tl.range)
- Per-subgroup Winsorized z-score (K=8, clip ±2.0)
- Rank-1 perturbation compression (vec_dim=2306, 28.8x compression)

---

## Current Best Hyperparameters

```python
HALF_POP = 7168          # antithetic pairs → pop=14336 (was 4096→8192)
SIGMA_START = 0.020      # perturbation scale (was 0.022, tuned for HALF_POP=7168)
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
| Triton, pop=8192, Adam, σ=0.02, nw=4, BK=32 | 2.44 | 11.4 | 175s | previous best |
| Triton, pop=8192, Adam, σ=0.02, nw=4, BK=32, FFN tl.range | 2.41 | 11.1 | 156s | previous best |
| Triton, pop=8192, Adam, σ=0.022, nw=4, BK=32, FFN+head tl.range | 2.49 | 12.1 | 153s | previous best |
| **Triton, pop=14336, Adam, σ=0.020, nw=4, BK=32, FFN+head tl.range** | **2.38** | **10.8** | **272s** | **current best (3-seed avg 2.376)** |

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
1. Pick a quality improvement from "What to Try Next — Quality Improvement"
2. Implement it
3. git add -A && git commit -m "description"
4. uv run train_eggroll.py --seed 42   (quick check: ~2.5 min)
5. Check: did val_loss DECREASE? Is time still ≤ 200s?
6. If promising (val_loss < 2.49): run 3-seed validation
   uv run validate.py                 (~25 min)
7. If 3-seed avg improves: git push, update this file, keep the change
8. If val_loss worse or time > 200s: git reset --hard HEAD~1
9. Go to step 1
```

Never stop to ask. The cost of a failed run is ~3 minutes (single seed). Keep experimenting.
Use single-seed tests for fast iteration, 3-seed only for promising results.

---

## Logging (results.tsv)

Tab-separated, one row per run, append-only.

Columns: `timestamp commit seed val_loss perplexity training_time_s peak_memory_mb status description`
