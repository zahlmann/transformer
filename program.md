# Fused Inference — Agent Program

*You are a GPU kernel engineer. Your job: build the fastest possible inference for this
small transformer using custom Triton kernels. The model is trained with standard backprop.
The focus is on learning to write high-performance GPU kernels for transformer inference.*

---

## Your Mission

1. Train the model with backprop (done: `uv run train_backprop.py`, val_loss=1.84)
2. Write a fused Triton inference kernel that generates text token-by-token (done: 4x speedup)
3. Benchmark against JAX/XLA baseline inference (done: 740 vs 185 tok/s)
4. Optimize: fuse operations, minimize memory bandwidth, maximize throughput

The goal is to learn kernel-writing skills on a small model where iteration is fast,
then apply those skills to larger models.

---

## Technology Stack

- **Triton** (OpenAI): Python-like GPU kernel language that compiles to PTX. Handles tiling,
  memory coalescing, and register allocation automatically.
- **JAX**: ML framework with XLA compiler. Used for model definition, training, and baseline.
- **jax-triton**: Bridge that calls Triton kernels from JAX with zero-copy tensor sharing.

---

## Model Architecture

```
Decoder-only transformer (character-level Shakespeare)
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128, vocab_size: 65
Parameters: 66,368
```

Weights: token_emb (65,64), pos_emb (128,64), Q/K/V/O (64,64 each), FFN up (64,256),
FFN down (256,64), LN scales/biases, output_proj (64,65).

---

## Current Kernel Infrastructure

### `kernels/fused_prefill.py` — Prefill Kernel

Fused Triton kernel that processes the ENTIRE forward pass in one kernel call:
Embedding → LayerNorm → Multi-Head Attention → FFN → LayerNorm → Output Projection

Key techniques:
- **All weights in registers.** Each weight matrix fits in register file. No HBM
  round-trips between layers.
- **bf16 tensor core matmuls with f32 accumulation.** Uses hardware tensor cores
  for 8-16x throughput vs scalar fp32.
- **K-tiled FFN loop.** The (128,256) FFN is tiled into 8 blocks of (128,32) using
  `tl.range(0, 256, 32)` with dynamic range to prevent register pressure from unrolling.
- **Causal attention in-register.** Full (128,128) attention matrix computed and softmaxed
  without going through HBM.
- **num_warps=4 (128 threads/block).** Optimal — fewer warps = more registers/thread.
  255 regs/thread, 0 bytes spill.
- **Outputs KV cache** for the decode phase.

### `kernels/fused_decode.py` — Decode Kernel

Single-token decode using KV cache. Uses element-wise ops (not tensor cores, since M=1).
Reads KV cache, computes attention over all past tokens, outputs logits + new K/V vectors.
JAX wrapper updates the cache via scatter.

---

## Results

```
Prompt: 64 tokens, Generate: 64 tokens

Triton fused:   740 tok/s  (86.5 ms)
JAX baseline:   185 tok/s  (362.6 ms)
Speedup:        4.0x
```

---

## Next Steps

### Phase A: Scale the Model

The current model is tiny (66K params, char-level, 1 layer). The fused kernel technique
works because everything fits in registers. Scaling up tests where this breaks and forces
new techniques.

**A1. BPE tokenization.** Switch from character-level (vocab=65) to subword tokens
(tiktoken or sentencepiece, vocab=512-8192). This immediately improves text quality
per parameter and is how real models work. The kernel needs a larger embedding table
and output projection — the output projection matmul (d_model × vocab) may no longer
fit in registers and will need tiling.

**A2. Bigger model.** Scale d_model (64→128→256), add layers (1→2→4), increase context
(128→512→1024). Key thresholds:
- d_model=128: weights ~260KB in bf16, exceeds register file. Need multi-block tiling.
- n_layers=2+: can't fuse everything into one kernel. Need one kernel per layer or
  a persistent kernel that loops over layers.
- context=512+: attention matrix (512×512 = 1MB) can't fit in registers. Need
  FlashAttention-style tiled attention.

**A3. More data.** TinyShakespeare (1MB) is too small for bigger models. Options:
- TinyStories (~500MB) — simple English, good for small models
- OpenWebText subset — real web text
- FineWeb-Edu subset — high-quality educational text

### Phase B: Kernel Optimizations

**B1. Quantization.** INT8 or FP8 weights for the decode kernel. Halves memory
bandwidth for weight loads. The decode kernel is memory-bound (loading KV cache +
weights), so this should give ~1.5-2x speedup.

**B2. Batched decode.** Process multiple sequences simultaneously. Each sequence
has its own KV cache but shares weights. The decode kernel becomes a (batch, d_model)
matmul instead of (1, d_model), which CAN use tensor cores.

**B3. Persistent decode kernel.** Keep one kernel running permanently, feed it tokens
via shared memory or global memory flags. Eliminates kernel launch overhead entirely
(currently ~7µs per decode step × 64 steps = ~0.5ms).

**B4. Speculative decoding.** Use the small model as a draft model: generate N candidate
tokens in one batch, then verify. If the verification model is the same (self-speculative),
this trades compute for latency.

### Phase C: Multi-Layer Fusion

For models with 2+ layers, the question is how to keep data in fast memory between layers:
- **Option 1: One kernel per layer.** Intermediate activations go through HBM between
  layers but stay in registers within each layer. Simple, composable.
- **Option 2: Shared memory handoff.** Write activations to shared memory at end of
  layer, read them back for next layer. Stays on-chip but limited to ~164KB.
- **Option 3: Persistent kernel.** One kernel that loops over layers, keeping activations
  in registers. Most complex but eliminates all intermediate memory traffic.

---

## Profiling Commands

```bash
uv run train_backprop.py             # train the model (4s)
uv run inference_benchmark.py        # inference speed comparison
```

---

## Files

```
program.md                          — this file (read first)
inference_guide.md                  — ground-up explanation of GPU kernels + this project
README.md                           — project overview
model.py                            — JAX transformer model (inference baseline)
train_backprop.py                   — standard backprop training
data.py                             — char-level Shakespeare dataset
kernels/fused_prefill.py            — fused Triton prefill kernel
kernels/fused_decode.py             — fused Triton decode kernel
inference_benchmark.py              — speed comparison benchmark
```

---

## Kernel Engineering Lessons

1. **num_warps=4 is optimal for this model.** Fewer warps = more registers per thread.
   num_warps=2 is slower (poor occupancy), num_warps=8 is slower (thread overhead).

2. **Dynamic loops (`tl.range`) prevent register-pressure blowup from unrolling.**
   The FFN K-tiling loop MUST use `tl.range`, not `tl.static_range`. Static unrolling
   of 8 iterations causes 340 bytes of register spilling.

3. **bf16 matmuls with f32 accumulation are the sweet spot.** FP8 was tried and was slower
   (register pressure from casts outweighed tensor core gains). fp16 out_dtype didn't help.

4. **The entire model fits in registers.** Total weight storage: ~66K params × 2 bytes =
   132KB. With 128 threads/block and 255 regs/thread, we have 128 × 255 × 4 = 131KB of
   register file. Tight but fits (weights are loaded on-demand, not all live simultaneously).

5. **Attention scores (128×128) in registers are the bottleneck.** This is 16K f32 values
   = 64KB. It works at num_warps=4 but is the largest live tensor.

6. **Decode can't use tensor cores.** With M=1, tensor cores need at least 16×16 tiles.
   Element-wise ops are faster for single-token decode.

7. **maxnreg doesn't help.** The kernel is compute-bound, not occupancy-bound. Forcing
   fewer registers just causes spilling.
