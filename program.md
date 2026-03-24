# EGGROLL Transformer — Agent Program

*You are an AI researcher. Your job: train a small decoder-only transformer using
EGGROLL (Evolution Strategies with low-rank perturbations) instead of backprop.
Make it as fast as possible while matching backprop accuracy. You work autonomously,
run experiments, log results, and keep going without stopping to ask for permission.*

---

## Current State (2026-03-24)

**Phase 1 (working): DONE.** EGGROLL trains a transformer from scratch.
**Phase 2 (quality): val_loss=2.67 vs backprop 2.45.** Gap=0.22. Close but not matched.
**Phase 3 (speed): 540s vs 1.3s.** 415x gap. No Triton kernel yet — main bottleneck.

Best EGGROLL config: `train_eggroll_optimized.py` with HALF_POP=1024, momentum=0.5,
alpha=0.50, bf16, orthogonal QR vectors, N_ACCUM=2.

---

## Files

```
program.md                    — this file (read first)
data.py                       — char-level Shakespeare dataset (65 vocab, 7842 train sequences)
model.py                      — decoder-only transformer (d=64, 1 layer, 2 heads, 66K params)
train_backprop.py             — backprop baseline (vanilla SGD, LR=3e-4, under-tuned)
train_eggroll.py              — fp32 EGGROLL (no momentum, no bf16 — superseded)
train_eggroll_optimized.py    — best EGGROLL (bf16, momentum, orthogonal vecs, accum)
benchmark.py                  — runs both and compares
```

---

## Results — Complete History

### Backprop baselines (10 epochs, same architecture, no warmup)

| LR | Decay | val_loss | ppl | Time | Notes |
|----|-------|----------|-----|------|-------|
| 3e-4 | 1.0 | 3.59 | 36.1 | 6.1s | initial under-tuned baseline |
| 1e-3 | 0.92 | 3.37 | 28.9 | 1.3s | first tuning |
| 2e-2 | 0.80 | 2.70 | 14.9 | 1.3s | what we initially compared against |
| 5e-2 | 0.90 | 2.54 | 12.7 | 1.3s | |
| 1e-1 | 0.90 | 2.50 | 12.2 | 1.3s | |
| **3e-1** | **0.90** | **2.45** | **11.6** | **1.3s** | **best backprop (ceiling)** |
| 5e-1 | 0.85 | 2.45 | 11.6 | 1.3s | saturated |

**Lesson: backprop with vanilla SGD tolerates VERY high LR (0.3!) for this tiny transformer.
Always sweep backprop LR aggressively (1e-4 to 1.0) before comparing. The initial LR=3e-4
was 1000x below optimal.**

### EGGROLL results (10 epochs, all bf16 unless noted)

| Config | val_loss | ppl | Time | Key change |
|--------|----------|-----|------|-----------|
| fp32, pop=512, alpha decaying | 3.94 | 51.4 | 148s | first working version |
| fp32, pop=2048, alpha=0.15 const | 3.70 | 40.6 | 501s | constant alpha discovery |
| **bf16**, pop=2048, alpha=0.20 | 2.83 | 17.0 | 270s | bf16 breakthrough |
| bf16, pop=2048, alpha=0.20, **momentum=0.5** | 2.80 | 16.5 | 270s | momentum eliminates oscillation |
| bf16, pop=2048, **alpha=0.50**, momentum=0.5 | 2.70 | 14.8 | 270s | high alpha sweet spot |
| bf16, pop=4096 (accum=2), alpha=0.50, mom=0.5 | 2.69 | 14.7 | 540s | gradient accumulation |
| bf16, pop=4096, alpha=0.50, mom=0.5, **ortho QR** | **2.67** | **14.4** | **540s** | **best EGGROLL** |

### Speed vs quality tradeoff (bf16, alpha=0.20, no momentum, no accum)

| Pop | val_loss | ppl | Time | Speed gap |
|-----|----------|-----|------|-----------|
| 64 | 3.74 | 42.2 | 16s | 12x |
| 128 | 3.25 | 25.9 | 25s | 19x |
| 256 | 3.05 | 21.2 | 43s | 33x |
| 512 | 2.95 | 19.0 | 79s | 61x |
| 2048 | 2.83 | 17.0 | 270s | 208x |

---

## What Worked (apply from day 1)

### 1. bf16 forward passes — the single biggest quality win (-0.87 loss)
Cast all perturbed params to bf16 before the forward pass. Keep perturbation math
and gradient estimation in fp32. The bf16 quantization noise acts as implicit
regularization for ES's already-noisy gradient estimates. **Does NOT help backprop**
(tested: backprop gets same val_loss in bf16 vs fp32). This is ES-specific.

### 2. Constant label smoothing, alpha=0.50 — second biggest win (-0.13 loss)
**Unlike MNIST where adaptive alpha decay was best**, transformers need CONSTANT
high alpha. Higher alpha = smoother fitness landscape = better ES gradient estimation.
The smoothing also provides strong regularization against overfitting.

Sweep results: alpha=0.10→2.86, 0.20→2.80, 0.25→2.77, 0.30→2.73, 0.35→2.72,
0.40→2.71, **0.50→2.70**, 0.60→2.70. Saturates around 0.50.

**Label smoothing HURTS backprop** (3.59→3.87 with alpha=0.20, T=2.0). This is
because backprop gets exact gradients — smoothing just makes the target less precise.
For ES, smoothing makes the landscape estimable with finite perturbations.

### 3. Momentum (beta=0.5) — eliminates oscillation (-0.03 loss)
ES gradient estimates have high variance. Momentum averages across batches, smoothing
out the noise. Without momentum, val_loss oscillates ±0.15 between epochs. With
momentum, it decreases monotonically.

**Momentum=0.9 explodes.** The effective LR with momentum beta is LR/(1-beta), so
beta=0.9 gives 10x effective LR. For ES with noisy gradients this is catastrophic.
Sweet spot: beta=0.5 (2x effective LR, manageable).

Note: momentum technically violates the MNIST "no cross-batch state" rule, but for
transformer training it's essential. The rule was designed for MNIST where ES gradients
were less noisy (higher pop/params ratio).

### 4. Per-layer LR scaling — stable from the start
Small params (biases, layer norms, <256 elements): 3.0x LR — they get excellent
gradient estimates because few parameters share few perturbation directions.
Small matrices (attention Q/K/V/O, <4096 elements): 1.5x LR.
Medium matrices (embeddings, output proj): 1.0x LR.
Large matrices (FFN up/down, >8192 elements): 0.7x LR — gradient quality is worst
for the largest layers.

### 5. Orthogonal perturbation vectors via QR — (-0.02 loss)
Instead of i.i.d. Gaussian random vectors, orthogonalize them:
```python
raw = jax.random.normal(key, (total_vec_dim, HALF_POP))
Q, _ = jnp.linalg.qr(raw)
vecs = Q.T * jnp.sqrt(total_vec_dim)
```
Since HALF_POP=1024 < total_vec_dim=2306, this gives 1024 perfectly orthogonal
vectors with zero redundancy. Standard Gaussian vectors have significant random overlap.
**Contradicts the MNIST finding** that "structured perturbations add JIT overhead" —
for transformers, the quality gain outweighs the small QR cost.

### 6. Temperature T=2.0 in the smoothed CE loss
Softens logits, makes the fitness landscape smoother. T=1.0 is too sharp (hard to
estimate gradient), T=3.0 is too flat (gradient signal too weak). T=2.0 is the sweet
spot (tested T=1.0, 2.0, 3.0).

### 7. Gradient accumulation (N_ACCUM=2)
Run 2 rounds of ES gradient estimation per batch with independent random vectors,
average the gradients, then apply one momentum+update step. Effective population
doubles (2048→4096) without increasing memory. Costs 2x time.
Diminishing returns: N_ACCUM=3 gave only 0.005 more improvement over N_ACCUM=2.

### 8. Rank-1 perturbation compression
For a (m,n) weight matrix, perturb with σ*outer(b,a) using m+n random values instead
of m*n. Total vec_dim=2306 vs 66K full params = 28.8x compression. This is the core
EGGROLL technique — confirmed it works for transformers (attention + FFN + embeddings).

### 9. Per-subgroup Winsorized z-score (from MNIST, works here too)
K=8 subgroups, clip ±2.0. Split fitness differences into groups, normalize each
independently, clip outliers. Same as MNIST — no need to change.

---

## What Did NOT Work for Transformer ES

### 1. Adaptive alpha decay (MNIST lesson that DOESN'T transfer)
MNIST used alpha starting at 0.30 decaying by 0.50 per epoch. For the transformer,
this causes val_loss to improve for 1-2 epochs then oscillate wildly as alpha drops.
The transformer needs CONSTANT alpha for consistent regularization throughout training.

### 2. Higher sigma (σ=0.06 vs σ=0.04)
Higher sigma makes gradient estimates noisier, not better. σ=0.04 gave val_loss=2.80,
σ=0.06 gave 2.90. Unlike the MNIST finding "higher sigma at lower pop," for
transformers the default σ=0.04 is already good.

### 3. High momentum (beta=0.9)
Effective LR becomes ~10x, causes divergence (val_loss→5.5). Beta=0.5 is the sweet
spot. Beta=0.6 gives similar results to 0.5.

### 4. Weight tying (output_proj = token_emb.T)
Saves 4160 params but HURTS quality for both backprop (3.59→3.83) and EGGROLL.
The embedding init scale (0.02) is wrong for the output projection role. Not worth
the complexity.

### 5. Vectorized base+correction forward pass
Tried restructuring the forward pass to compute base x@W once and apply rank-1
corrections per perturbation across the chunk dimension. XLA could NOT optimize the
complex computation graph efficiently — 3x SLOWER (765s training, 287s JIT) than
the straightforward vmap approach. The extra dimensions (chunk, batch, seq, d_model)
and einsum operations create worse memory access patterns.

### 6. Nested lax.scan for entire epoch
Wrapping the batch loop in a lax.scan (so one JIT compiles the whole epoch) gave
NO speed improvement (263s vs 265s). Python dispatch overhead for 610 batch
iterations is negligible (~1ms total). But JIT time increased from 5s to 32s because
of the larger computation graph. **Not worth it.**

### 7. Larger POP_CHUNK (32 or 64 instead of 16)
More perturbations vmapped simultaneously = more memory pressure. POP_CHUNK=32 and
64 were both SLOWER than POP_CHUNK=16. The GPU can't efficiently process 32+
simultaneous transformer forward passes due to memory bandwidth saturation.

### 8. Temperature T=3.0
Too smooth — the gradient signal becomes too weak. val_loss=3.14 vs 2.70 with T=2.0.
The fitness landscape becomes nearly flat and ES can't estimate useful gradients.

### 9. Low alpha with momentum (alpha=0.10, momentum=0.5)
Momentum alone isn't enough regularization. Still need high alpha for smooth fitness
landscape. alpha=0.10 with momentum: 2.86. alpha=0.50 with momentum: 2.70.

---

## Differences from MNIST That Matter

| MNIST | Transformer | Why |
|-------|------------|-----|
| Adaptive alpha decay | Constant alpha=0.50 | Transformer overfits with decaying regularization |
| No momentum | Momentum beta=0.5 essential | Higher gradient noise needs temporal smoothing |
| Pop=1680 for 118K params (70:1 ratio) | Pop=2048 for 66K params (32:1 ratio) | Attention creates more complex loss landscape |
| Fused Triton kernel = main speedup | No Triton kernel yet | Attention makes fusion much harder |
| Structured perturbations hurt JIT | Orthogonal QR helps quality | Quality is bottleneck, not JIT time |
| 1.44x backprop time | 415x backprop time | Without kernel fusion, ES can't compete on speed |

---

## What to Try Next (Remaining Opportunities)

### Quality improvements (to close the 0.22 gap to backprop)
1. **Within-batch guided perturbations**: Split population — use first 256 perturbations
   for rough gradient, bias remaining 768 toward that direction. Single-batch version
   of Guided ES (no cross-batch state needed). Research suggests this can significantly
   improve gradient quality.
2. **Higher population**: Pop=8192 or 16384. Each doubling costs ~2x time but should
   give diminishing-returns quality improvement. We went 2048→4096 and got -0.02.
3. **Scale up architecture**: d_model=128 or 2 layers. More params but also more
   expressive — might close the gap if the bottleneck is model capacity, not ES quality.
4. **Adam-like adaptive LR**: Maintain per-parameter running variance of ES gradients,
   use it to scale updates (like Adam's second moment). This is cross-batch state but
   momentum already crosses that line.
5. **Cosine LR schedule**: Tested briefly, similar results to exponential. Worth
   trying more carefully with the current momentum+alpha setup.

### Speed improvements (to close the 415x gap)
1. **Fused Triton kernel for FFN+CE**: The FFN (64→256→64) + output (64→65) + CE loss
   is structurally identical to the MNIST 3-layer MLP kernel. This is the most tractable
   Triton kernel to write. The attention part stays in JAX.
2. **Full Triton kernel**: Fuse the entire forward pass including attention. Extremely
   complex — attention creates (128,128) intermediate per head that doesn't fit in
   registers. Would need tiled attention (Flash Attention style) within the kernel.
3. **Reduce population with better gradients**: Guided ES or adaptive methods could
   maintain quality at lower pop, directly reducing forward pass count.
4. **fp8 for matmuls inside bf16 forward pass**: Use fp8 tensor cores for the linear
   layers (Q/K/V, FFN, output proj). Keeps perturbation math in bf16/fp32.

### Experimental ideas (uncertain payoff)
- Learned perturbation directions (train a small network to generate perturbation vectors)
- Sparse perturbation for embeddings (only perturb embedding rows that appear in the batch)
- Different loss functions (InfoNCE, contrastive, mutual information)
- Curriculum learning (easier sequences first)

---

## Architecture Details

```
Model: decoder-only transformer
d_model: 64
n_heads: 2 (d_head=32)
n_layers: 1
d_ff: 256 (4x d_model)
context_len: 128
vocab_size: 65 (character-level)
Parameters: 66,368
Perturbation vec_dim: 2306 (28.8x compression)
```

Parameter breakdown:
- token_emb (65, 64) = 4,160
- pos_emb (128, 64) = 8,192
- layer0.ln1 scale+bias = 128
- layer0.attn Q/K/V/O (64, 64) × 4 = 16,384
- layer0.ln2 scale+bias = 128
- layer0.ffn.up (64, 256) + bias = 16,640
- layer0.ffn.down (256, 64) + bias = 16,448
- ln_final scale+bias = 128
- output_proj (64, 65) = 4,160

Data: tiny Shakespeare, 7842 train sequences × 128 tokens, 871 val sequences.

---

## Best EGGROLL Hyperparameters

```python
HALF_POP = 1024         # antithetic pairs -> 2048 per round
POP_CHUNK = 16          # vmap chunk size (16 is optimal for RTX 4080 SUPER)
N_ACCUM = 2             # gradient accumulation -> effective pop=4096
SIGMA_START = 0.04      # perturbation scale
SIGMA_DECAY = 0.998     # per epoch
LR_START = 0.012        # learning rate
LR_DECAY = 0.92         # per epoch (exponential)
ALPHA = 0.50            # label smoothing (CONSTANT, no decay)
TEMPERATURE = 2.0       # CE temperature scaling
MOMENTUM = 0.5          # SGD momentum for noisy ES gradients
N_SUBGROUPS = 8         # Winsorized z-score groups
CLIP_RANGE = 2.0        # z-score clipping
# Orthogonal perturbation vectors via QR decomposition
# Per-layer LR scaling: 3x small, 1.5x attn, 1.0x medium, 0.7x FFN
```

---

## Setup

1. `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read this file and `../eggroll_mnist/mnist_eggroll_optimized.py`
3. Set up the project: `uv init --bare && uv add jax jaxlib numpy`
4. Implement backprop baseline first (establishes the accuracy target)
5. Implement EGGROLL training
6. Begin the experiment loop

**Important:** Always commit and push work to the autoresearch branch regularly.

---

## Experiment Loop

```
1. Pick an idea (architecture change, hyperparameter, kernel optimization)
2. Implement it
3. git add -A && git commit -m "description" && git push
4. uv run benchmark.py > run.log 2>&1
5. Check results: loss, time, memory
6. If crashed: fix and retry
7. If improved: keep. If not: git reset --hard HEAD~1
8. Log to results.tsv
9. Go to step 1
```

Never stop to ask if you should continue. The cost of a failed run is seconds.

---

## Logging (results.tsv)

Tab-separated, one row per run, NOT committed to git.

Columns: `commit\tloss\tperplexity\ttraining_time_s\tpeak_memory_mb\tstatus\tdescription`
