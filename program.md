# Fused Inference — Agent Program

*You are a GPU kernel engineer. Your job: build the fastest possible inference for this
small transformer using custom Triton kernels. The model is trained with standard backprop.
The focus is on learning to write high-performance GPU kernels for transformer inference.*

**IMPORTANT: Commit and push after every meaningful step. Don't batch up changes.**

---

## Mission

Build the fastest possible inference for this transformer using custom Triton kernels.

### Completed

1. Fused Triton prefill + decode kernels (single kernel for small model)
2. BPE tokenization + tiled output projection (vocab=1024, zero overhead)
3. Multi-block prefill for d_model=128 (BLOCK_SEQ=32, 3 kernels/layer)
4. Fused multi-layer decode (all layers + output in one kernel)
5. Host-side optimization: in-kernel cache updates, precomputed bf16 weights

### Current Performance

Small (d=64, 1L): 3056 tok/s, 16.9x over JAX
Large (d=128, 2L): 2589 tok/s, 13.9x over JAX

### Next: scale the model and data (see YOUR NEXT TASK below)

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

### What was done (optimization)

- ~~**Persistent kernel**~~ DONE — fused 2-layer decode into single kernel (35% speedup)
- ~~**Precompute weights**~~ DONE — avoid per-step dtype conversion (57% speedup)
- ~~**In-kernel cache updates**~~ DONE — eliminate .at[].set() (3.6x speedup, biggest win)
- ~~**Multi-layer fusion**~~ DONE — h stays in registers between layers
- **Quantization**: NOT NEEDED at this scale (dispatch-bound, not bandwidth-bound)

---

## YOUR NEXT TASK: Scale Up

The kernel infrastructure is solid (14-17x speedup). Now scale up the model and data
to make the inference kernels work harder and the text quality actually good.

### Step 1: Bigger Training Data

Shakespeare alone (1.1M chars) is too small for models beyond ~200K params — they overfit.
Get more data to justify bigger models:

- **Option A: TinyStories** (~500M tokens of simple English stories). Good for small models.
  Available at https://huggingface.co/datasets/roneneldan/TinyStories
- **Option B: OpenWebText subset** (~8M tokens sample). More diverse, harder to learn.
- **Option C: Combine Shakespeare + other public domain literature** (Project Gutenberg).
  Keep the style consistent. Maybe 10-50M chars total.

Train the BPE tokenizer on the combined corpus. Vocab=2048-4096 should work well with
more data (the current 1024 was chosen because Shakespeare alone is too small).

### Step 2: Bigger Model (d_model=256, 4 layers)

With more data, scale the model:
- d_model=256, n_heads=8, d_head=32, n_layers=4, d_ff=1024
- ~5-10M parameters (vs current 674K)
- context_len=256 (or 512 if FlashAttention is implemented)

This forces real kernel engineering:
- **Prefill**: BLOCK_SEQ=16 (256 d_model overflows BLOCK_SEQ=32 blocks).
  h_block = (16, 256) = 16KB. Attention = (16, 256) = 16KB. Fits.
- **Decode**: still trivial (M=1, h is 1KB). Fused multi-layer kernel just gets longer.
  May need to pass weights via a packed buffer instead of 50+ individual pointers.
- **Output projection**: with vocab=4096, 32 tiles of 128. Already works.

### Step 3: FlashAttention for context_len=512+

At context=256 with BLOCK_SEQ=16, attention scores are (16, 256) = 16KB per block. Fits.
At context=512, scores are (16, 512) = 32KB. Still fits but tight with other live tensors.
At context=1024+, need FlashAttention (tiled attention with online softmax).

FlashAttention approach:
```
for kv_start in range(0, seq_len, KV_TILE):
    # Load Q block (already have from current block)
    # Load K, V tile from cache
    # Compute partial attention scores
    # Update running softmax (online normalization)
    # Accumulate weighted V
```

This keeps attention scores small: (BLOCK_SEQ, KV_TILE) instead of (BLOCK_SEQ, seq_len).

### Step 4: Training Quality

With bigger data and model:
- Add learning rate warmup + cosine decay schedule
- Add weight decay (AdamW)
- Train for more epochs (20-50)
- Target val_loss < 3.0 with BPE (coherent multi-sentence text)
- Consider gradient accumulation if batch doesn't fit in memory

### Step 5: Benchmark at Scale

Compare Triton vs JAX at the larger scale. The speedup story changes:
- With d_model=256, the model is compute-heavier → kernel fusion matters more
- With context=512, attention is O(n²) → FlashAttention is critical
- With 4 layers, the fused decode kernel saves 3 extra launches per layer

### What NOT to change

- Keep the same technology stack (Triton + JAX + jax-triton)
- Keep BPE tokenization (trained on corpus)
- Keep the multi-block prefill architecture (just reduce BLOCK_SEQ)
- Keep the fused multi-layer decode approach (just add more layers)
- Keep the in-kernel cache update optimization

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
kernels/fused_decode_2layer.py      — fully fused 2-layer decode (fastest path)
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
