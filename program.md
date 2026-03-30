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
6. TinyStories dataset (14M tokens) + trained BPE tokenizer (vocab=4096)
7. Scaled to d_model=256, 4 layers, 8 heads (5.3M params, ppl=5.40)
8. Fused N-layer decode with packed weights/caches (any layer count)
9. FlashAttention for context>256 (tiled KV + online softmax)
10. AdamW + LR warmup + cosine decay training improvements
11. Speculative decoding with parallel verification kernel
12. Scaled to d=512, 8 layers, 16 heads (29.7M params, ppl=2.91)
13. Multi-SM decode kernel: grid=(N_HEADS,) with atomic barriers (5.2x speedup, 287→1519 tok/s)

### Current Performance (RTX 4080 Super)

```
XL model (d=512, h=16, l=8, ctx=512, 29.7M params, ppl=2.91):
  Decode (multi-SM):   1734 tok/s (0.58 ms/tok)  ← 6x speedup via grid=(16,)
  Decode (single-SM):  287 tok/s  (3.49 ms/tok)   (previous baseline)
  Prefill (128 tok):   6.1 ms    (21,168 tok/s)
  Weight buffer:       59.3 MB   (bf16)
  KV cache:            8.4 MB    (bf16, per sequence)
  Bandwidth util:      14%       (of 836 GB/s theoretical, was 2%)

Previous model sizes:
  d=64,  1L:    3056 tok/s
  d=128, 2L:    2589 tok/s
  d=256, 4L:    1396 tok/s
  d=512, 8L:     287 tok/s  → 1734 tok/s (multi-SM, 2742 w/o sync)
```

**Key finding: multi-SM decode (grid=16) gave 6x speedup.** The single-SM kernel
left 79 of 80 SMs idle. Splitting attention heads across blocks with atomic barriers
parallelizes both attention and FFN. Bandwidth utilization improved from 2% to 14%.
Without int() sync: 2742 tok/s (9.6x). Theoretical minimum is 0.081 ms/tok — 7x headroom.

---

## Technology Stack

- **Triton** (OpenAI): Python-like GPU kernel language that compiles to PTX. Handles tiling,
  memory coalescing, and register allocation automatically.
- **JAX**: ML framework with XLA compiler. Used for model definition, training, and baseline.
- **jax-triton**: Bridge that calls Triton kernels from JAX with zero-copy tensor sharing.

---

## Model Architecture

```
Decoder-only transformer
d_model: 64-512, n_heads: 2-16 (d_head=32), n_layers: 1-8, d_ff: 4*d_model
context_len: 128-512

Small:  d=64,  h=2,  l=1, vocab=65,   params=66,368     (Shakespeare, char)
Medium: d=64,  h=2,  l=1, vocab=1024, params=189,120    (Shakespeare, BPE)
Large:  d=128, h=4,  l=2, vocab=1024, params=674,304    (Shakespeare, BPE)
XL:     d=256, h=8,  l=4, vocab=4096, params=5,318,144  (TinyStories, BPE)
XXL:    d=512, h=16, l=8, vocab=4096, params=29,660,160 (TinyStories full, BPE)
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

## Results (Optimization History)

```
Phase A1: d=64, 1L, vocab=1024 (Shakespeare):
  Initial:                          740 tok/s
  + tiled output projection:        758 tok/s  (zero overhead from tiling)

Phase A2: d=128, 2L, vocab=1024 (Shakespeare):
  Multi-block (3 launches/step):    335 tok/s
  + fused decode (1 launch/step):   454 tok/s  (+35%)
  + precomputed bf16 weights:       713 tok/s  (+57%)
  + in-kernel cache updates:       2504 tok/s  (+3.6x, biggest win)

Phase A3: d=256, 4L, vocab=4096, 5.3M params (TinyStories):
  Fused N-layer decode:            1396 tok/s

Phase A4: d=512, 8L, vocab=4096, 29.7M params (TinyStories full):
  Fused N-layer decode:             287 tok/s  (3.48 ms/tok, 2% BW utilization)

Phase A5: Multi-SM decode optimization:
  Profiling revealed: kernel=93%, host=3%, argmax=4% of step time
  Multi-SM decode (grid=16):       1734 tok/s  (0.58 ms/tok, 14% BW utilization)
  Without int() sync:              2742 tok/s  (0.36 ms/tok)
  Speedup vs single-SM:              6.0x (with sync), 9.6x (without sync)

Key findings:
- Tiled output projection has ZERO overhead vs register-only.
- Python dispatch overhead per jt.triton_call is ~0.4ms. Fusing all decode
  into one kernel eliminated 128 launches (35% speedup).
- .astype(bf16) per decode step created 1764 unnecessary allocations.
  Precomputing once gave another 57% speedup.
- In-kernel KV cache updates were the biggest single win (3.6x).
- At d=512 single-SM, bandwidth utilization was only 2%. GPU was 98% idle.
- Multi-SM decode with atomic barriers: 5.2x speedup (287→1519 tok/s).
  Splits attention heads across 16 blocks, distributes FFN across all blocks.
  Uses release/acquire atomics for cross-block synchronization.
  Bandwidth utilization improved from 2% to 12%.
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

## COMPLETED: Scale Up (Phase A3)

All 5 steps completed. Speedup holds at 15x with 8x larger model.

### What was done

1. **Bigger Training Data**: TinyStories dataset (55M chars, 14M BPE tokens, vocab=4096).
   HuggingFace ByteLevel BPE tokenizer trained on the corpus. Zero UNK rate.

2. **Bigger Model**: d=256, h=8, l=4, d_ff=1024, ctx=256, 5.3M params.
   BLOCK_SEQ=16 in prefill (fits (16,256)=16KB h_block in registers).
   Fused N-layer decode with packed weight buffer + packed KV caches.

3. **FlashAttention**: Tiled KV (KV_TILE=64) with online softmax for context>256.
   Keeps scores at (BLOCK_SEQ, KV_TILE) = (16, 64) = 4KB instead of (16, 512) = 32KB.

4. **Training Quality**: AdamW (wd=0.1) + LR warmup (200 steps) + cosine decay.
   val_loss=1.69, ppl=5.40 in 20 epochs. Coherent multi-paragraph stories.

5. **Benchmark**: 1396 tok/s Triton vs 92 tok/s JAX = 15.1x speedup.

### Key engineering decisions

- **Packed weight buffer**: All per-layer weights concatenated into one bf16 buffer.
  Kernel computes offsets from layer index. Avoids 50+ individual pointer arguments.
- **Packed KV caches**: All layers' K/V caches in one flat buffer. Stays packed between
  decode steps — no per-step pack/unpack overhead.
- **tl.static_range for layers**: Loop unrolled at compile time (4 iterations).
  Compiler can reuse registers between layers. Fast for ≤8 layers.
- **BLOCK_SEQ as kernel parameter**: Not module-level constant. Allows different
  tile sizes per model (32 for d=128, 16 for d=256).

---

## COMPLETED: Speculative Decoding (Phase B)

Implemented speculative decoding with a custom parallel verification kernel.

### What was done

1. **Draft model**: Trained d=128, h=4, l=1 (1.3M params, ppl=8.76) on TinyStories
   with same vocab=4096 as target. Saved to `draft_weights.pkl`.

2. **Speculative decode algorithm**: Draft generates K tokens autoregressively,
   target verifies in parallel, greedy accept/reject with first-disagreement cutoff.

3. **Parallel verification kernel** (`kernels/verify_decode.py`): Processes K draft
   tokens through the full target model in ONE kernel call. Key techniques:
   - Pads to PAD_K=16 internally for `tl.dot` shape compatibility (inner dim >= 16)
   - Online softmax over tiled KV cache (same as FlashAttention) for prefix tokens
   - Causal attention among draft tokens with runtime `real_k` mask
   - Single grid=(1,) launch — all K tokens in one thread block

4. **Correctness verified**: Parallel kernel matches sequential decode within bf16
   tolerance (max logit diff < 0.06).

### Results

```
Standard target decode:                1395 tok/s
Speculative K=2 (sequential verify):    707 tok/s  (0.51x)  acceptance=51%
Speculative K=2 (parallel verify):      715 tok/s  (0.51x)  acceptance=50%
Speculative K=4 (sequential verify):    488 tok/s  (0.35x)  acceptance=37%
Speculative K=4 (parallel verify):      632 tok/s  (0.45x)  acceptance=36%

Parallel kernel speedup over sequential: 1.30x at K=4
```

### Key findings

- **Speculative decoding has diminishing returns at small model scale.**
  With fused kernels, both draft (~2500 tok/s) and target (~1400 tok/s) are
  extremely fast. The speed ratio is only ~2x, not the ~10x needed for
  speculative decoding to be profitable.

- **Acceptance rate is the bottleneck.** With a 1-layer draft (ppl=8.76) vs
  4-layer target (ppl=5.40), greedy acceptance is only 36-51%. Speculative
  decoding needs 80%+ acceptance to break even at this speed ratio.

- **The parallel verification kernel works.** 1.3x faster than K sequential
  decode calls at K=4. The kernel correctly handles prefix cache attention,
  causal draft-draft attention, and online softmax — all in one launch.

- **When speculative decoding helps:** Large, slow target models (e.g., 70B at
  20 tok/s) with a fast draft (7B at 200 tok/s) give a 10x speed ratio.
  At that ratio, even 50% acceptance yields 2-3x speedup.

### Kernel engineering lesson

23. **Batch verification is a "mini-prefill" with existing KV cache.** The
    verification kernel processes K tokens through all layers, attending to P
    cached tokens (via tiled KV + online softmax) plus K draft tokens (via
    causal intra-draft attention). Padding K to 16 satisfies Triton's
    `tl.dot` minimum inner dimension requirement.

---

## COMPLETED: Scale Up (Phase A4)

Scaled to d=512, 8 layers, 29.7M params on full TinyStories (487M tokens, 1.9GB).

### What was done

1. **Full TinyStories**: Downloaded all 2.1M stories (1.9GB). Chunked BPE tokenization
   (50MB chunks) to avoid OOM — encoding 1.9GB at once used 46GB+ RAM. Retrained BPE
   tokenizer on full dataset.

2. **Training**: d=512, h=16, l=8, d_ff=2048, ctx=512, 29.7M params.
   batch=16, lr=1e-4, 3 epochs (~4.4 hours). val_loss=1.068, ppl=2.91.
   Data kept on CPU, batches streamed to GPU to avoid data OOM.

3. **Kernel adaptations for d=512** (register pressure is the main challenge):
   - **BLOCK_SEQ=8** for prefill (h_block = (8,512) = 16KB fits in registers)
   - **tl.dot for projections**: element-wise (512,32) intermediates overflow registers
     (128 regs/thread > 255 limit). tl.dot tiles internally, avoiding materialization.
   - **Tiled KV decode**: (512,32) full cache load overflows shared memory.
     Added online softmax with KV_TILE=64, same technique as FlashAttention.
   - **Smaller output VTILE=32**: (512,128) weight load = 128KB > 130KB register file.
     Reduced to (512,32) = 64KB.

4. **Benchmark**: 287 tok/s, 3.48 ms/tok, 2% bandwidth utilization.

### Key engineering decisions

- **tl.dot vs element-wise at d=512**: At d≤256, element-wise `tl.sum(h[:, None] * W, axis=0)`
  works because (256,32) = 64 regs/thread fits alongside h (2 regs) in 255 budget. At d=512,
  (512,32) = 128 regs alone, product = another 128 → 260 > 255. tl.dot avoids materializing
  the full intermediate by tiling the reduction internally.
- **Triton [0,:] indexing not supported**: Used `.sum(axis=0)` on (1,N) results from tl.dot.
- **Chunked tokenization**: HuggingFace tokenizer uses ~25x text size in working memory.
  1.9GB text → 46GB+ RAM → OOM killed. 50MB chunks cap peak at ~12GB.

---

## YOUR NEXT TASK: Further Scaling or Optimization

**GPU: RTX 4080 Super (Ada Lovelace, 16GB VRAM, 101KB shared memory, 52 TFLOPS FP16, 836 GB/s bandwidth)**

### How to measure progress

Run `uv run profile_kernels.py` after any kernel change. The hard metrics to beat:

```
CURRENT (d=512, 8L, 29.7M params, multi-SM grid=16):
  Decode throughput:     1753 tok/s  (with int sync for token collection)
  Decode no-sync:        ~2900 tok/s (in-kernel argmax, deferred collection)
  Decode latency:        0.57 ms/tok (with sync), 0.35 ms/tok (no sync)
  Bandwidth utilization: 14% of 836 GB/s
  Prefill latency:       6.3 ms (128 tokens)
  Theoretical min:       0.075 ms/tok (63 MB unique data @ 836 GB/s)

PREVIOUS BASELINE (single-SM, grid=1):
  Decode throughput:     287 tok/s (3.48 ms/tok, 2% BW)
```

The 2% bandwidth utilization means the GPU is 98% idle — there is massive room
for improvement. Any optimization that moves bandwidth utilization upward is real progress.

### Step 0: Profile to understand where time is spent — COMPLETED

Profiled with custom host overhead measurement (profile_host_overhead.py).

```
PROFILING RESULTS (2026-03-30):
  Total per step:       3.598 ms  (281 tok/s)
  GPU kernel time:      3.351 ms  (93% of total)
  argmax + GPU→CPU sync: 0.134 ms  (4%)
  Host overhead:         0.113 ms  (3%)

  Without int() sync:   3.031 ms/step (330 tok/s, 17% faster)
  Amortized (batch):    3.028 ms/step (330 tok/s)
  Theoretical min:      0.081 ms/tok (836 GB/s)
```

**Key findings:**
1. **Host overhead is NOT the bottleneck** — only 3% of step time.
   The original hypothesis was wrong. Python/JAX dispatch is negligible at d=512.
2. **The GPU kernel itself is 93% of step time** (3.35ms of 3.6ms).
3. **The kernel runs on 1 SM** (grid=(1,)). The 4080 Super has 80 SMs.
   79 SMs are completely idle during decode. This is the #1 bottleneck.
4. **Removing int() sync gives 17% speedup** (281→330 tok/s) by avoiding
   GPU→CPU transfer per step. Easy win.
5. **Single-SM bandwidth**: 67.7MB / 3.35ms = 20.2 GB/s achieved.
   This is ~2x the per-SM DRAM bandwidth (836/80 = 10.4 GB/s),
   indicating significant L2 cache hits (weights are 59.3MB, L2 is 64MB).

**Revised priority based on profiling:**
- Step 1 (host overhead) → low impact, only 3% of time
- Step 2b (multi-SM decode) → HIGHEST IMPACT, 79 idle SMs
- Step 1c (device argmax) → moderate impact, 17% speedup

### Step 1: Eliminate host overhead — COMPLETED (low impact)

Profiling showed host overhead is only 3% (0.11ms). Not the bottleneck at d=512.
**1c. In-kernel argmax** — DONE. Avoids GPU→CPU sync in decode loop (1592→2900 tok/s).

### Step 2: Multi-SM decode — COMPLETED (biggest win)

**2a. num_warps sweep** — DONE. num_warps=4 and 8 are tied (2872 vs 2915 tok/s).
Not warp-limited.

**2b. Multi-SM decode** — DONE. grid=(N_HEADS=16,) with atomic barriers.

**2b. Multi-SM decode** — Current decode launches grid=(1,) — one thread block on
one SM. The 4080 Super has 80 SMs. 79 SMs are completely idle during decode.
Possible approaches:
- Split heads across SMs: grid=(N_HEADS,), each block handles one head's attention
- Split layers across SMs: pipeline layers across blocks with barriers
- Split the KV tiling across SMs: each block handles a tile of the cache

This is the most impactful architectural change. Going from 1 SM to even 8 SMs
would give up to 8x speedup if the work parallelizes well.

**2c. Batched inference** — Process B sequences in parallel. With M>1:
- Projections become real matmuls: tl.dot((B, D_MODEL), (D_MODEL, D_HEAD))
- Tensor cores activate (need M >= 16 for full utilization)
- Each SM has more work → better utilization
- Trade-off: increases per-sequence latency slightly, but total tok/s goes up dramatically
- Target: 4-8x throughput at batch=8-16

### Step 3: Reduce memory traffic per step

Achieved: 0.35 ms/tok (no sync), 14% BW utilization. Remaining gap: 4.7x vs theoretical.

**Bottleneck analysis (2026-03-31):**
- Unique data per step: 63 MB. Theoretical at 836 GB/s: 0.075 ms.
- Achieved kernel time: 0.57 ms (block_until_ready), 0.35 ms (amortized/pipelined).
- 17 barriers per step × 5-10µs each = 0.085-0.17 ms (15-30% of kernel time).
- Weights (59.3 MB) barely fit in L2 (64 MB). With KV cache (8.4 MB), L2 overflows.
- num_warps sweep: 2/4/8/16 all within 5% — not warp-limited.
- Only 16 blocks on 80 SMs — low occupancy limits memory request parallelism.

**3a. Reduce barriers** — Merge FFN reduction with next layer's LN1. Use persistent
accumulators. Target: 9 barriers instead of 17. Saves 0.04-0.08 ms.

**3b. More blocks (higher grid)** — grid=(32,) or grid=(64,) to use more SMs and
increase outstanding memory requests. Requires splitting heads across multiple blocks
for KV attention (2 blocks per head, each handling half the KV tiles).

**3c. INT8 weight quantization** — Halve weight buffer to ~30 MB. Fits comfortably
in L2 with KV cache (38 MB). Effective bandwidth is L2 bandwidth (~2 TB/s), not
DRAM. Theoretical: 38 MB / 2000 GB/s = 0.019 ms. Requires dequantization in kernel.

**3d. GQA (Grouped-Query Attention)** — 4 KV heads for 16 Q heads → 4x less KV
cache traffic. Requires retraining but reduces 8 MB KV cache to 2 MB.

**3e. Persistent kernel across steps** — Kernel loops internally across all decode
steps, polling for new tokens. Eliminates all per-step launch overhead and workspace
allocation. Most ambitious approach.

### What NOT to change

- Keep the same technology stack (Triton + JAX + jax-triton)
- Keep BPE tokenization (trained on corpus)
- Keep the multi-block prefill architecture
- Keep the fused multi-layer decode approach
- Keep the in-kernel cache update optimization

---

## Profiling Commands

```bash
uv run profile_kernels.py                   # primary benchmark (run after every change)
uv run profile_kernels.py --detailed        # per-component breakdown
uv run inference_benchmark.py               # quick throughput check

# Nsight Compute for detailed GPU metrics
/usr/local/cuda/bin/ncu --set full uv run profile_kernels.py

# Nsight Systems for timeline view
/usr/local/bin/nsys profile -t cuda uv run profile_kernels.py

# Training
uv run train_backprop.py --d-model 512 --n-heads 16 --n-layers 8 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16
```

---

## Files

```
program.md                          — this file (read first)
repo_explained_from_zero.md                  — ground-up explanation of GPU kernels + this project
README.md                           — project overview
model.py                            — JAX transformer model (inference baseline)
train_backprop.py                   — AdamW training with LR schedule
data.py                             — Shakespeare + TinyStories (char, GPT-2 BPE, trained BPE)
kernels/fused_prefill.py            — fused Triton prefill kernel (d_model≤64)
kernels/fused_decode.py             — fused Triton decode kernel (d_model≤64)
kernels/block_prefill.py            — multi-block prefill + FlashAttention (d_model≥128)
kernels/block_decode.py             — per-layer decode + orchestrator (d_model≥128)
kernels/fused_decode_2layer.py      — fully fused 2-layer decode
kernels/fused_decode_nlayer.py      — fully fused N-layer decode (packed weights/caches)
kernels/multi_sm_decode.py          — multi-SM decode: grid=(N_HEADS,) with atomic barriers
profile_kernels.py                  — primary profiling tool (run after every change)
profile_host_overhead.py            — detailed host vs GPU timing breakdown
inference_benchmark.py              — quick throughput benchmark
baseline_metrics.txt                — current numbers to beat
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

18. **Packed weight buffers scale to any layer count.** With 4+ layers, passing individual
    weight pointers becomes unwieldy (50+ arguments). Concatenating all per-layer weights
    into one bf16 buffer and computing offsets from layer index is cleaner and equally fast.

19. **Packed KV caches eliminate per-step allocation.** Keep all layers' K/V caches in one
    flat buffer between decode steps. The kernel reads from kv_in and writes to kv_out
    directly. No Python-side pack/unpack between steps.

20. **15x speedup holds across model scales.** d=64/1L: 16.9x, d=128/2L: 14.0x,
    d=256/4L: 15.1x. The kernel fusion advantage is consistent — it's not just a
    small-model artifact. As models grow, compute increases but so does the benefit
    of eliminating host overhead.

21. **BLOCK_SEQ must scale inversely with d_model.** Register file is fixed at ~127KB.
    With d=128: BLOCK_SEQ=32 → h=(32,128)=16KB. With d=256: BLOCK_SEQ=16 → h=(16,256)=16KB.
    The h_block size stays at 16KB by halving BLOCK_SEQ when doubling d_model.

22. **FlashAttention is only needed for context>256.** At context=256 with BLOCK_SEQ=16,
    scores are (16,256)=16KB per head — fits alongside K(32KB)+V(32KB)+accumulators.
    At context=512+, the full scores matrix overflows. Tiled KV with online softmax
    (KV_TILE=64) keeps peak at ~20KB per tile regardless of context length.

24. **At d=512, element-wise projections overflow registers.** Each (D_MODEL, D_HEAD) =
    (512, 32) intermediate uses 128 registers per thread (out of 255 max). The product
    `h[:, None] * W` needs another 128 → 260 > 255, causing spills to shared memory.
    Fix: use `tl.dot` which tiles the reduction internally. With `tl.dot((1, 512) @ (512, 32))`,
    Triton never materializes the full (512, 32) intermediate.

25. **Output projection needs smaller VTILE at d=512.** Loading (D_MODEL, VOCAB_TILE) =
    (512, 128) bf16 = 128KB ≈ entire register file. Reduced to VTILE=32 → (512, 32) = 32KB.
    4x more loop iterations but avoids register spilling.

26. **Tiled KV attention in decode is essential for MAX_SEQ=512.** Loading the full
    (MAX_SEQ, D_HEAD) = (512, 32) cache at once overflows shared memory (139KB > 101KB).
    Online softmax with KV_TILE=64 tiles the cache: each tile loads (64, 32) = 4KB.
    Same algorithm as FlashAttention but for the M=1 single-token case.

27. **At d=512, the GPU kernel is 93% of step time, not host overhead.** Profiling
    showed: kernel=3.35ms (93%), argmax+sync=0.13ms (4%), host=0.11ms (3%). The
    original hypothesis that Python dispatch was the bottleneck was wrong at this
    model scale. Host overhead matters for small models (d≤128) but is negligible
    at d=512 where the kernel dominates.

28. **Multi-SM decode with atomic barriers gives 5.2x speedup.** grid=(N_HEADS=16,)
    instead of grid=(1,) — each block handles one attention head, and all blocks
    split the FFN. Uses `tl.atomic_add` with `sem='release'/'acquire', scope='gpu'`
    for cross-block barriers. Two barriers per layer: one after attention (before
    FFN), one after FFN (before next layer). All blocks redundantly compute LN and
    reductions (32KB from L2, negligible cost). 287→1519 tok/s, 2%→12% BW utilization.

29. **Redundant computation is cheaper than synchronization.** In the multi-SM kernel,
    all 16 blocks independently compute LayerNorm and reduce the 16 partial results
    (16×512 f32 = 32KB reads). This is redundant but costs only ~1µs per block from
    L2 cache. The alternative — having one block compute and broadcast — would need
    an additional barrier (~5µs) plus the serial bottleneck.

30. **Use tl.range (not static_range) for the layer loop in large kernels.** With 8
    layers and the full multi-SM kernel body (attention + FFN + barriers + reductions),
    `tl.static_range` unrolls the loop at compile time, creating enormous IR that takes
    10+ minutes to compile. `tl.range` compiles the loop body once and executes it
    dynamically — identical runtime performance, 10x faster compilation.
