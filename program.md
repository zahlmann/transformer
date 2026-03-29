# Fused Inference — Agent Program

*You are a GPU kernel engineer. Your job: build the fastest possible inference for this
small transformer using custom Triton kernels. The model is trained with standard backprop.
The focus is on learning to write high-performance GPU kernels for transformer inference.*

**IMPORTANT: Commit and push after every meaningful step. Don't batch up changes.**

---

## Your Mission

1. Train the model with backprop (done: char val_loss=1.84, BPE val_loss=3.77)
2. Write a fused Triton inference kernel that generates text token-by-token (done: 4x speedup)
3. Benchmark against JAX/XLA baseline inference (done: 758 vs 174 tok/s)
4. BPE tokenization + tiled output kernel (done: vocab=1024, 8 tiles, 4.35x speedup)
5. Optimize: fuse operations, minimize memory bandwidth, maximize throughput

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
Decoder-only transformer (Shakespeare)
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128

Small:  d=64,  h=2, l=1, vocab=65,   params=66,368
Medium: d=64,  h=2, l=1, vocab=1024, params=189,120  (BPE)
Large:  d=128, h=4, l=2, vocab=1024, params=674,304  (BPE, multi-block)
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
- **Tiled output projection.** Vocab dimension tiled in VOCAB_TILE=128 chunks via
  `tl.range(0, VOCAB_PAD, VOCAB_TILE)`. Each tile independently computes and stores
  logits to HBM. VOCAB_SIZE/VOCAB_PAD passed as kernel `tl.constexpr` parameters.
- **Outputs KV cache** for the decode phase.

### `kernels/fused_decode.py` — Decode Kernel

Single-token decode using KV cache. Uses element-wise ops (not tensor cores, since M=1).
Reads KV cache, computes attention over all past tokens, outputs logits + new K/V vectors.
Same tiled output projection as prefill. JAX wrapper updates the cache via scatter.

---

## Results

```
Char-level (vocab=65, register-only output):
  Triton fused:   740 tok/s  (86.5 ms)
  JAX baseline:   185 tok/s  (362.6 ms)
  Speedup:        4.0x

BPE (vocab=1024, tiled output — 8 tiles of 128):
  Triton fused:   758 tok/s  (84.5 ms)
  JAX baseline:   174 tok/s  (367.4 ms)
  Speedup:        4.35x

Phase A2: d_model=128, n_heads=4, n_layers=2, vocab=1024, 674K params:
  Multi-block (3 launches/step):   335 tok/s,  1.89x
  + fused decode (1 launch/step):  454 tok/s,  2.58x
  + precomputed bf16 weights:      713 tok/s,  3.95x
  + in-kernel cache updates:      2504 tok/s, 13.98x
  JAX baseline:                    179 tok/s

Key findings:
- Tiled output projection has ZERO overhead vs register-only.
- Multi-block approach (BLOCK_SEQ=32) enables any d_model.
- Python dispatch overhead per jt.triton_call is ~0.4ms. Fusing all decode
  into one kernel eliminated 128 launches (35% speedup).
- .astype(bf16) per decode step created 1764 unnecessary allocations.
  Precomputing once gave another 57% speedup.
- Text quality greatly improved with 2-layer model.
```

---

## COMPLETED: BPE Tokenization + Tiled Output Kernel

Switched from char-level (vocab=65) to BPE subword tokens (vocab=1024).
This was the first scaling step — forces tiled output projection in the kernel.

### What was done

1. **BPE tokenizer**: Trained on Shakespeare using HuggingFace `tokenizers` (ByteLevel
   BPE). GPT-2's tiktoken had 16-25% UNK rate on Shakespeare (11K unique tokens in corpus).
   The trained tokenizer has 0% UNK with vocab=1024.

2. **Tiled output projection**: Both kernels now tile the output projection in VOCAB_TILE=128
   chunks using `tl.range(0, VOCAB_PAD, VOCAB_TILE)`. VOCAB_SIZE and VOCAB_PAD are kernel
   constexpr parameters, so jax-triton JIT-compiles per vocab configuration.

3. **Results**: Zero overhead from tiling (4.35x speedup vs 4.0x baseline). The output
   projection is compute-light — attention dominates the kernel runtime. Text output is
   coherent Shakespeare-like dialogue.

### Design decisions taken

- **Tokenizer**: GPT-2 tiktoken had 24-46% UNK rate on Shakespeare (11K unique tokens).
  Trained a BPE tokenizer directly on the corpus using HuggingFace `tokenizers` — 0% UNK.
- **Vocab size**: 1024. Gives 8 tiles of 128, meaningful tiling work. Both 512 and 1024
  produce similar bits-per-character (~2.2) — the model's capacity is the bottleneck.
- **VOCAB_TILE**: 128 (same as FFN BLOCK_K). Each tile loads (64, 128) bf16 weights,
  produces (128, 128) f32 logits (prefill) or (128,) f32 logits (decode).
- **d_model stays at 64**: model is parameter-inefficient (embedding tables dominate)
  but the kernel engineering goal was achieved without changing the core architecture.

---

## COMPLETED: Phase A2 — Bigger Model

Scaled d_model 64→128, added 2 layers, 4 attention heads. Used multi-block approach:
3 Triton kernels per layer (proj, attn, ffn) + 1 output kernel, grid=(4,) with
BLOCK_SEQ=32. Per-block register peak: ~52KB (vs 127KB budget).

### Block kernels architecture

```
kernels/block_prefill.py:
  _proj_kernel:   LN + Q/K/V projections   (grid=4, each block=32 rows)
  _attn_kernel:   causal attention + O + residual
  _ffn_kernel:    LN + FFN + residual
  _output_kernel: final LN + tiled output projection

kernels/block_decode.py:
  _decode_layer_kernel: per-layer decode (M=1, trivial register pressure)
  _decode_output_kernel: final LN + output projection
```

### Phase A2 remaining (not yet done)

- context=512+: attention matrix needs FlashAttention-style tiled attention
- d_model=256+: even BLOCK_SEQ=32 blocks overflow registers, need smaller blocks

### Phase B: Kernel Optimizations

- ~~**Persistent kernel**: eliminate launch overhead~~ DONE — fused 2-layer decode
  into single kernel (3→1 launch/step, 35% speedup)
- ~~**Precompute weights**: avoid per-step dtype conversion~~ DONE — 57% speedup
- **Quantization**: INT8/FP8 weights for decode (memory-bound, ~1.5-2x speedup)
- **Batched decode**: multiple sequences share weights, enables tensor cores
- **Speculative decoding**: trade compute for latency

### Phase C: Multi-Layer Fusion — DONE

Fused 2-layer decode kernel keeps h in registers between layers (no HBM round-trip).
Prefill still uses separate kernels per stage (HBM between stages) due to inter-block
synchronization requirements.

---

## Profiling Commands

```bash
uv run train_backprop.py                                          # char-level (4s)
uv run train_backprop.py --tokenizer trained_bpe --bpe-vocab 1024 # BPE (6s)
uv run inference_benchmark.py                                     # inference speed
```

---

## Files

```
program.md                          — this file (read first)
inference_guide.md                  — ground-up explanation of GPU kernels + this project
README.md                           — project overview
model.py                            — JAX transformer model (inference baseline)
train_backprop.py                   — standard backprop training
data.py                             — Shakespeare dataset (char, GPT-2 BPE, trained BPE)
kernels/fused_prefill.py            — fused Triton prefill kernel (d_model≤64)
kernels/fused_decode.py             — fused Triton decode kernel (d_model≤64)
kernels/block_prefill.py            — multi-block prefill kernels (d_model≥128)
kernels/block_decode.py             — per-layer decode + orchestrator (d_model≥128)
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

8. **Tiled output projection has zero overhead.** Going from register-only (vocab=65)
   to tiled (vocab=1024, 8 tiles of 128) maintained or improved speedup (4.0x → 4.35x).
   The output projection is not the bottleneck — attention is. Tiling is free.

9. **VOCAB_SIZE/VOCAB_PAD as kernel constexpr parameters.** Using `tl.constexpr` function
   parameters (not module-level) lets jax-triton JIT-compile a kernel variant per vocab
   size. Clean separation between kernel logic and model configuration.

10. **Multi-block tiling is the key to scaling d_model.** With d_model=128, h is (128,128) =
    64KB — can't fit with attention scores (also 64KB) in 127KB register file. Tiling
    the sequence into BLOCK_SEQ=32 row blocks reduces h to (32,128) = 16KB, scores to
    (32,128) = 16KB. Peak per block: ~52KB. Scales to any d_model.

11. **Projections and attention need separate kernel launches.** Within a single Triton
    kernel, blocks execute in parallel with no ordering guarantee. Block 3's attention
    can't read K/V written by block 0 in the same launch. Split into proj_kernel (writes
    K/V to HBM) → attn_kernel (reads K/V from HBM).

12. **Decode kernel scales trivially with d_model.** M=1 means all tensors are 1D (d_model,).
    With d_model=128, h is only 512 bytes. A single fused kernel per layer works for any
    d_model. The bottleneck is attention over the KV cache: (max_seq, d_head) per head.

13. **HBM traffic between kernels costs ~2x speedup** initially. Went from 4.3x to 1.9x.
    But fusing decode back into one kernel + precomputing weights recovered most of it
    (1.9x → 3.9x). The real cost was Python dispatch, not HBM.

14. **Python dispatch dominates small-model latency.** Each jt.triton_call has ~0.4ms of
    Python/JAX overhead. With 3 calls per decode step × 63 steps = 189 calls → 75ms of
    pure overhead. Fusing into 1 call/step saved 35%.

15. **Never convert dtypes inside the decode loop.** `.astype(bf16)` creates a new JAX
    array every call. With 28 weight tensors × 63 steps = 1764 allocations → 57% slowdown.
    Precompute bf16 weights once before the loop.

16. **Fold KV cache updates into the kernel.** `.at[:, pos, :].set()` creates a new array
    copy every call. 4 caches × 0.27ms = 1.09ms per step — 73% of decode time!
    Having the kernel write full updated caches directly gave 3.6x speedup.

17. **For small models, host overhead > GPU compute.** The GPU kernel takes 0.4ms per
    decode step, but Python/JAX dispatch, dtype conversions, and array scatters added
    1.1ms. Eliminating all host-side overhead gave 7.5x improvement (335→2504 tok/s).
