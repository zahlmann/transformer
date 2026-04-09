# Custom GPU Kernels for Transformer Inference

How we fuse an entire 24-layer, 306M-parameter transformer decode step into a single
GPU kernel launch — and why that makes it fast. This document explains every line of
inference code in the project, from first principles. You only need to know Python.

**Files covered:**
- `kernels/fused_decode_nlayer.py` — weight/KV packing utilities + single-SM decode kernel
- `kernels/multi_sm_decode.py` — multi-SM fused decode kernel (the fast one)
- `generate.py` — streaming text generation CLI
- `profile_kernels.py` — benchmarking and roofline analysis

---

## Table of Contents

1. [How GPUs Work](#1-how-gpus-work)
2. [GPU Memory Hierarchy](#2-gpu-memory-hierarchy)
3. [The Memory Wall](#3-the-memory-wall)
4. [Triton: GPU Programming in Python](#4-triton-gpu-programming-in-python)
5. [Our Transformer Model](#5-our-transformer-model)
6. [Prefill and Decode](#6-prefill-and-decode)
7. [Weight Packing](#7-weight-packing)
8. [KV Cache Packing](#8-kv-cache-packing)
9. [The Single-SM Decode Kernel](#9-the-single-sm-decode-kernel)
10. [The Multi-SM Decode Kernel](#10-the-multi-sm-decode-kernel)
11. [The Generation Loop](#11-the-generation-loop)
12. [Profiling and Roofline Analysis](#12-profiling-and-roofline-analysis)

---

## 1. How GPUs Work

A GPU is a chip with thousands of tiny processors that all run the same program
simultaneously. Understanding four concepts is enough to understand our kernels.

### Streaming Multiprocessors (SMs)

The RTX 4080 Super has **80 SMs** (Streaming Multiprocessors). Each SM is like a
small independent computer with its own registers, shared memory, and execution units.
When you launch a GPU program (called a "kernel"), the GPU assigns blocks of work
to SMs. If you launch 16 blocks, 16 SMs each run one block simultaneously.

### Warps and Threads

Each SM runs threads in groups of **32 called warps**. All 32 threads in a warp
execute the same instruction at the same time (this is called SIMT — Single Instruction,
Multiple Threads). If one thread in a warp takes a branch and another doesn't, both
paths execute sequentially — the divergent threads just sit idle. This is why GPU
code avoids if/else when possible.

Our kernels use `num_warps=4`, meaning each block runs 4 × 32 = **128 threads**.
With 128 threads, each thread gets access to more registers (the GPU has a fixed
register file per SM that's divided among active threads).

### Tensor Cores

Modern NVIDIA GPUs have special hardware units called **tensor cores** that can
multiply small matrices (like 16×16 × 16×16) in a single clock cycle. When Triton
sees a `tl.dot(A, B)` call, it automatically uses tensor cores if the shapes are
compatible. This gives us massive throughput for matrix multiplications — the
RTX 4080 Super can do **52 TFLOPS** of FP16 matrix math per second using tensor cores.

For comparison, without tensor cores, the same GPU does about 26 TFLOPS — tensor
cores roughly double matrix multiplication throughput.

### How a Kernel Launch Works

When Python calls a GPU kernel, here's what happens:

1. The CPU sends a small command to the GPU: "run this program with this many blocks"
2. The GPU scheduler assigns each block to an SM
3. All blocks run simultaneously (if there are enough SMs)
4. When all blocks finish, the CPU can read the results

The key insight: launching a kernel has **fixed overhead** (~0.1-0.4ms for Python dispatch
via `jax_triton`). If the kernel itself only takes 0.5ms, that overhead is significant.
This is why we fuse everything — embedding, 24 transformer layers, and output projection
— into a **single kernel launch**. One launch, one overhead cost.

**Why is this unusual?** Standard frameworks like PyTorch/JAX run each operation
(normalization, projection, attention, FFN) as a separate kernel launch. For a 24-layer
model, that's hundreds of launches per token — each with dispatch overhead and each
writing intermediate results to global memory and reading them back. Our approach
keeps the hidden state `h` in registers across all 24 layers. It's never written to
HBM between layers. This is only possible because we write custom Triton kernels
instead of composing framework-provided operations.

---

## 2. GPU Memory Hierarchy

The GPU has four levels of memory, each faster but smaller than the last. Understanding
this hierarchy is the key to writing fast kernels.

```
┌──────────────────────────────────────────────────────┐
│                    Registers                          │
│   Speed: ~20 TB/s    Size: 255 per thread             │
│   Scope: private to each thread                       │
│   Access: instant (0 cycles latency)                  │
├──────────────────────────────────────────────────────┤
│                  Shared Memory                        │
│   Speed: ~3-6 TB/s   Size: 101 KB per SM              │
│   Scope: shared within one block (all threads on SM)  │
│   Access: ~20 cycles latency                          │
├──────────────────────────────────────────────────────┤
│                    L2 Cache                            │
│   Speed: ~3-6 TB/s   Size: 64 MB total                │
│   Scope: shared across all SMs                        │
│   Access: ~200 cycles latency                         │
├──────────────────────────────────────────────────────┤
│               HBM (Global Memory)                     │
│   Speed: 836 GB/s    Size: 16 GB total                │
│   Scope: accessible by CPU and GPU                    │
│   Access: ~400 cycles latency                         │
└──────────────────────────────────────────────────────┘
```

### Registers

Each thread has up to **255 registers**, each holding 4 bytes (one float32 or two
bfloat16 values). With 128 threads per block (our `num_warps=4`), that's
128 × 255 × 4 = **131 KB** of register file per block. Registers are the fastest
storage — zero latency, bandwidth in the tens of terabytes per second.

The challenge: if your kernel needs more than 255 registers per thread, the compiler
"spills" excess values to slower shared memory or global memory. This is called
**register pressure** and it's a constant concern in our kernels.

### Shared Memory

Each SM has **101 KB** of shared memory (on our RTX 4080 Super). All threads in a
block can read and write to it. It's much faster than global memory (~20 cycles vs
~400 cycles latency) but much smaller.

We use shared memory as a scratch pad — for example, when computing RoPE (rotary
position embeddings), we write intermediate values to shared memory so we can read
them back in a different order. Triton manages shared memory automatically when you
use `tl.dot` — it stages matrix tiles in shared memory for tensor core consumption.

**Key constraint:** When Triton computes `tl.dot(A, B)`, it loads tiles of A and B
into shared memory. If A is (512, 64), that's 512 × 64 × 2 = 64 KB in bf16. Two
such tiles would be 128 KB > 101 KB, so we can't load two large matrices at once.
This drives our `PROJ_TILE=512` design — we tile projections to fit within shared memory.

### L2 Cache

The **64 MB** L2 cache sits between the SMs and main memory. When any SM reads from
global memory, the data passes through L2 and gets cached. If another SM (or the same
SM later) reads the same address, it hits L2 instead of going to slow HBM.

For our 306M model, the weight buffer is **607 MB** — almost 10x larger than L2.
This means weights can't stay cached between decode steps. Every step re-fetches
all weights from HBM. At smaller model sizes (d=512, ~55 MB weights), the entire
model fits in L2, giving much better performance.

We use **L2 cache eviction hints** (`eviction_policy='evict_last'` and `'evict_first'`)
to tell the GPU which data to keep vs evict. KV cache data is reused across steps,
so we hint `evict_last` (keep it). Output projection data is used once, so we hint
`evict_first` (evict it immediately to make room for more useful data).

### HBM (High Bandwidth Memory)

The main GPU memory: **16 GB** at **836 GB/s** bandwidth. This is where all weights,
KV caches, and activations live. Despite the name "high bandwidth," 836 GB/s is the
bottleneck for inference — the compute units can process data much faster than HBM
can deliver it.

---

## 3. The Memory Wall

Here's the central insight for understanding inference performance:

**Generating one token requires reading all model weights from memory, but only doing
a tiny amount of math with each weight.**

Let's do the arithmetic for our 306M model:

```
Weight buffer:    607 MB (all 24 layers' weights, bf16)
KV cache:         ~6 MB (grows with sequence length)
Total per step:   ~613 MB of data to read

HBM bandwidth:    836 GB/s
Theoretical min:  613 MB ÷ 836 GB/s = 0.73 ms per token

Compute required: ~612M multiply-adds per token
Tensor core throughput: 52 TFLOPS
Compute time:     612M ÷ 52T = 0.012 ms per token
```

The compute takes **0.012 ms** but reading the data takes **0.73 ms**. The GPU spends
98% of its time waiting for data, not computing. This is the **memory wall** —
inference is **bandwidth-bound**, not compute-bound.

This has a profound consequence: **the only way to make inference faster is to read
less data or read it faster**. Making the math more efficient doesn't help — the GPU
is already idle most of the time. This is why we:

1. Pack everything into bf16 (half the bytes of f32)
2. Fuse all layers into one kernel (each weight is read once, used, and discarded)
3. Use L2 eviction hints (keep useful data cached, evict one-time-use data)
4. Don't bother with quantization (our goal is learning kernels, not shrinking models)

**Bandwidth utilization** is our key metric. Our multi-SM kernel achieves about 17%
of theoretical bandwidth, delivering ~231 tok/s. The gap between 17% and 100% comes
from:
- Barrier synchronization overhead (blocks waiting for each other)
- Uneven work distribution across blocks
- Memory access patterns that don't achieve peak bandwidth
- L2 cache misses forcing HBM round-trips

---

## 4. Triton: GPU Programming in Python

[Triton](https://triton-lang.org/) is a Python-like language for writing GPU kernels.
It compiles to the same low-level GPU instructions (PTX) as CUDA, but handles memory
coalescing, register allocation, and tiling automatically. You write at the level of
**blocks of data** rather than individual threads.

### Key Triton Concepts Used in Our Code

**`@triton.jit`** — Decorator that compiles a Python function into a GPU kernel:
```python
@triton.jit
def my_kernel(ptr, N: tl.constexpr):
    ...
```

**`tl.constexpr`** — Marks a parameter as a compile-time constant. The compiler sees
the actual value and can optimize accordingly. All our shape parameters (D_MODEL,
N_HEADS, etc.) are constexpr because they never change at runtime.

**`tl.program_id(0)`** — Returns the index of the current block in the grid. If you
launch with `grid=(16,)`, blocks get IDs 0 through 15. Each block runs independently
on its own SM.

**`tl.arange(0, N)`** — Creates a vector of consecutive integers [0, 1, 2, ..., N-1].
**N must be a power of 2** — this is a Triton requirement. We use this to create index
vectors for loading/storing data:
```python
d = tl.arange(0, D_BLOCK)  # [0, 1, 2, ..., 1023]
h = tl.load(ptr + d)       # loads 1024 consecutive values
```

**`tl.load(ptr + offsets, mask=mask, other=0.0)`** — Reads values from GPU memory at
the given addresses. The `mask` parameter controls which lanes are active — masked-off
lanes get the `other` value instead of reading memory. We use masks for two purposes:
1. **D_MODEL padding:** when D_MODEL=1024 but D_BLOCK=1024 (both happen to match here,
   but the mask handles the general case)
2. **Sequence masking:** in attention, we mask out future positions (tile_pos > pos)

**`tl.store(ptr + offsets, values, mask=mask)`** — Writes values to GPU memory.

**`tl.dot(A, B)`** — Matrix multiplication using tensor cores. A and B must be 2D with
compatible shapes. Returns A @ B. Triton handles tiling and shared memory staging
automatically. We use bf16 inputs with f32 accumulation for numerical precision:
```python
result = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16)).to(tl.float32)
```

**`tl.range(start, stop, step)`** — A dynamic loop. The compiler generates a single
loop body that executes repeatedly. This is crucial for our 24-layer kernel — if we
used `tl.static_range`, the compiler would unroll all 24 iterations, duplicating the
code 24 times. This causes massive register spill and 10+ minute compilation times.
`tl.range` compiles the body once.

**`tl.static_range(start, stop, step)`** — A compile-time unrolled loop. Each iteration
becomes separate code. We use this for small loops (like iterating over KV_SPLITS=1-2)
where the overhead of loop control would exceed the body.

**`tl.atomic_add(ptr, val, sem='acq_rel', scope='gpu')`** — Atomically adds `val` to
the value at `ptr` and returns the old value. We use this for **cross-block barriers**
— all blocks increment a counter, and the last one to arrive knows everyone is done.
The `sem` (semantics) and `scope` parameters control memory ordering visibility.

**`grid=(N,)`** — When launching a kernel, the grid specifies how many blocks to run.
`grid=(16,)` launches 16 blocks, each getting a unique `tl.program_id(0)` from 0 to 15.

**`num_warps=4`** — How many warps per block. 4 warps × 32 threads = 128 threads.
More warps mean more threads sharing the register file (fewer registers per thread).
4 is optimal for our kernels — 2 has poor occupancy, 8 causes register pressure.

**`num_stages=1`** — Pipeline depth for memory loads. `num_stages=2` would double-buffer
tiles (loading the next tile while computing the current one), but this requires double
the shared memory. At D_BLOCK=1024, double-buffering would need 128 KB > 101 KB shared
memory, so we use `num_stages=1`.

### bf16 Computation with f32 Accumulation

Throughout our kernels, we follow this pattern:
```python
# Load in bf16 (saves memory bandwidth — half the bytes of f32)
w = tl.load(ptr).to(tl.bfloat16)
h = h_norm.to(tl.bfloat16)

# Multiply in bf16 on tensor cores (fast, uses specialized hardware)
result = tl.dot(h[None, :], w)

# Accumulate in f32 (prevents precision loss over many additions)
accum += result.to(tl.float32)
```

bf16 (bfloat16) has the same exponent range as f32 but only 7 bits of mantissa
(vs 23 for f32). Individual multiplications are fine, but summing many bf16 values
loses precision. So we multiply in bf16 (fast, on tensor cores) and accumulate in
f32 (accurate, on regular ALUs).

**Why bf16 specifically?** bf16 was designed by Google Brain specifically for deep
learning. Unlike fp16 (IEEE half-precision), bf16 keeps the same 8-bit exponent as
f32, so it handles the same range of magnitudes — no overflow/underflow surprises.
The tradeoff is fewer mantissa bits (7 vs 10 in fp16), but this is fine for neural
network weights. This "mixed precision" approach — bf16 for compute, f32 for
accumulation — is universal in LLM training and inference since 2020. It halves
memory bandwidth (the bottleneck for decode) while maintaining numerical accuracy.

---

## 5. Our Transformer Model

Our model is a **decoder-only transformer** with 306M parameters, using the modern
"Llama-style" architecture that has become the standard recipe since 2023. Each
component is a conscious choice over alternatives:

| Component | Replaces | Why |
|-----------|----------|-----|
| RMSNorm | LayerNorm | 10-15% faster, same quality |
| RoPE | Learned pos embeddings | Relative positions, no extra params, length generalization |
| GQA (4 KV heads) | MHA (16 KV heads) | 4x smaller KV cache, <1% quality loss |
| SwiGLU | ReLU/GELU FFN | ~15% better quality per FLOP |
| No biases | Biases | Simpler, no quality impact |
| Tied embeddings | Separate output head | Fewer params, good regularizer at 306M scale |

These are not experimental — they are the converged standard used by Llama 1/2/3/4,
Mistral, Qwen, DeepSeek, and Gemma as of 2026. Each subsection below explains what
the component does, why it exists, and exactly how it's implemented in our kernels.

Here are the key specs:

```
d_model  = 1024    # hidden dimension: every token is a vector of 1024 numbers
n_heads  = 16      # attention heads: 16 independent attention patterns per layer
n_kv_heads = 4     # KV heads: only 4 unique key/value sets (GQA, explained below)
d_head   = 64      # per-head dimension: 1024 ÷ 16 = 64 numbers per head
n_layers = 24      # depth: the input passes through 24 transformer layers
d_ff     = 2816    # FFN hidden dimension: SwiGLU expands from 1024 → 2816 → 1024
context_len = 512  # maximum sequence length
vocab_size = 32000 # vocabulary: 32K BPE tokens
```

### What Each Layer Does

Every transformer layer takes a 1024-dimensional vector (the hidden state `h`) and
transforms it through two sub-layers:

**Sub-layer 1: Attention**
```
h_norm = RMSNorm(h)          # normalize to unit variance
Q, K, V = h_norm @ Wq/Wk/Wv # project into query/key/value spaces
Q, K = RoPE(Q, K, pos)       # inject position information
attn = softmax(Q @ K.T) @ V  # weighted sum of values
h = h + attn @ Wo            # residual connection
```

**Sub-layer 2: Feed-Forward Network (SwiGLU)**
```
h_norm = RMSNorm(h)                      # normalize again
gate = h_norm @ W_gate                    # gate projection (1024 → 2816)
up = h_norm @ W_up                        # up projection (1024 → 2816)
ffn_out = (SiLU(gate) * up) @ W_down      # activate, multiply, project down (2816 → 1024)
h = h + ffn_out                           # residual connection
```

After all 24 layers, a final RMSNorm and output projection produce logits:
```
h = RMSNorm(h)
logits = h @ token_embedding.T   # tied embeddings: reuse the input embedding matrix
```

### RMSNorm

RMSNorm normalizes a vector by dividing by the root-mean-square of its elements,
then scaling by a learned parameter:

$$\text{RMSNorm}(h) = \gamma \cdot \frac{h}{\sqrt{\frac{1}{d}\sum\_{i=1}^{d} h\_i^2 + \epsilon}}$$

In code:
```python
h_norm = scale * h * rsqrt(mean(h * h) + 1e-5)
```

Concrete example with a 4-element vector:
```
h = [2.0, -1.0, 0.5, 1.5]
scale = [1.1, 0.9, 1.0, 0.8]

mean(h²) = (4.0 + 1.0 + 0.25 + 2.25) / 4 = 1.875
rsqrt(1.875 + 0.00001) = 1 / sqrt(1.875) ≈ 0.730

h_norm = scale * h * 0.730
       = [1.1 * 2.0 * 0.730,  0.9 * -1.0 * 0.730,  ...]
       = [1.606, -0.657, 0.365, 0.876]
```

Unlike LayerNorm, RMSNorm doesn't subtract the mean — it only divides by the RMS.
This is ~10-15% faster (one less reduction operation) and works just as well in practice.

**Why RMSNorm?** The original transformer (2017) used LayerNorm, which both
re-centers (subtracts mean) and re-scales (divides by std). Zhang & Sennrich
(2019) showed that the re-centering is unnecessary — only the re-scaling matters
for training stability. Removing it saves one reduction operation per normalization
(a parallel sum across 1024 elements). RMSNorm was adopted by Llama 1 (2023) and
is now universal in all major LLMs (Llama 2/3/4, Mistral, Qwen, DeepSeek, Gemma).
No alternative has emerged as of 2026.

### RoPE (Rotary Position Embedding)

RoPE encodes position information by rotating pairs of dimensions in Q and K by
angles that depend on position. Each pair of dimensions rotates at a different
frequency — low dimensions rotate slowly (capturing broad position patterns),
high dimensions rotate fast (capturing fine-grained position).

The rotation formula for position $p$ and dimension pair $i$:

$$\theta\_i = 10000^{-2i/d\_{\text{head}}}$$

$$\begin{bmatrix} q'\_{2i} \\\ q'\_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(p\theta\_i) & -\sin(p\theta\_i) \\\ \sin(p\theta\_i) & \cos(p\theta\_i) \end{bmatrix} \begin{bmatrix} q\_{2i} \\\ q\_{2i+1} \end{bmatrix}$$

In our implementation, the "even/odd" split is actually "first half / second half":
```python
q_lo, q_hi = Q[:d_head//2], Q[d_head//2:]
Q_rotated_lo = q_lo * cos - q_hi * sin
Q_rotated_hi = q_lo * sin + q_hi * cos
```

The RoPE tables are precomputed once in `precompute_rope_table()` (`model.py:101-107`):
```python
def precompute_rope_table(context_len, d_head, base=10000.0):
    half = d_head // 2                              # 32 for d_head=64
    freqs = base ** (-arange(0, half) * 2.0 / d_head)  # 32 frequencies, from 1.0 to ~0.00015
    positions = arange(context_len)                  # [0, 1, 2, ..., 511]
    angles = positions[:, None] * freqs[None, :]     # (512, 32) angle matrix
    return cos(angles), sin(angles)                  # each (512, 32)
```

Concrete example for d_head=64, position 5:
```
freqs = [10000^0, 10000^(-2/64), 10000^(-4/64), ..., 10000^(-62/64)]
      = [1.0, 0.724, 0.524, ..., 0.000147]

angles at pos 5 = [5*1.0, 5*0.724, 5*0.524, ..., 5*0.000147]
               = [5.0, 3.62, 2.62, ..., 0.000735]

cos_val = [cos(5.0), cos(3.62), ..., cos(0.000735)]
sin_val = [sin(5.0), sin(3.62), ..., sin(0.000735)]
```

The key property: the dot product Q·K between two positions depends only on their
**relative distance**, not absolute positions. This means the model generalizes to
positions it hasn't seen during training.

**Why RoPE?** There are three generations of position encoding:

1. **Sinusoidal (original transformer, 2017):** Fixed sine/cosine patterns added
   to the input embeddings. The position signal gets diluted through layers.
2. **Learned embeddings (GPT-2, 2019):** A separate trainable embedding per
   position. Hard-caps context length — can't generalize beyond training length.
3. **RoPE (Su et al. 2021):** Rotation applied to Q and K at every layer. No
   extra parameters. Relative-position-aware by construction.

RoPE's advantage over both: it's applied at every layer (position info can't be
forgotten), requires zero additional parameters, and enables length generalization
with extensions like NTK-aware scaling or YaRN for models with 128K+ context
windows. Adopted by Llama 1 (2023), RoPE is now the universal default for
decoder-only transformers in 2026.

### GQA (Grouped-Query Attention)

Standard multi-head attention has separate K and V matrices for each head. With 16
heads and d_head=64, that's 16 × 64 = 1024 dimensions for K and 1024 for V.

GQA (Grouped-Query Attention) **shares K and V across groups of heads**. Our model
has 16 query heads but only 4 KV heads. Each group of 4 query heads shares one KV head:

```
Query heads:  0  1  2  3 | 4  5  6  7 | 8  9 10 11 | 12 13 14 15
KV heads:     0  0  0  0 | 1  1  1  1 | 2  2  2  2 |  3  3  3  3

GQA_GROUP = N_HEADS // N_KV_HEADS = 16 // 4 = 4
kv_head = head_id // GQA_GROUP    # maps query head → shared KV head
```

**Why GQA?** Standard multi-head attention (MHA) gives each query head its own K
and V — maximum quality but the KV cache is large. Multi-Query Attention (MQA,
Shazeer 2019) goes to the opposite extreme: all heads share one K and one V —
minimum cache but measurable quality loss. GQA (Ainslie et al. 2023) is the
sweet spot: group query heads and share K/V within each group.

Our 4 KV heads for 16 query heads cuts KV cache by 4x with <1% quality loss:

```
Full MHA KV cache:  16 heads × 512 positions × 64 dims × 2 bytes = 1.0 MB/layer
GQA KV cache:        4 heads × 512 positions × 64 dims × 2 bytes = 0.25 MB/layer
With 24 layers:      24 MB → 6 MB total KV cache (4x reduction)
```

GQA has been the standard since Llama 2 (2023). DeepSeek V2/V3 introduced MLA
(Multi-head Latent Attention), which compresses KV via low-rank projections —
MLA is gaining traction in frontier models (Kimi K2.5, GLM-5) but GQA remains
the default for most new models, especially at smaller scales.

### SwiGLU FFN

SwiGLU is a gated activation function that replaced the standard ReLU FFN:

$$\text{SwiGLU}(h) = (\text{SiLU}(h W\_{\text{gate}}) \odot (h W\_{\text{up}})) W\_{\text{down}}$$

where SiLU(x) = x × sigmoid(x) and ⊙ is element-wise multiplication.

```python
gate = h_norm @ W_gate    # (1024,) → (2816,) — gate projection
up   = h_norm @ W_up      # (1024,) → (2816,) — up projection
act  = SiLU(gate) * up    # (2816,) — gated activation
out  = act @ W_down        # (2816,) → (1024,) — down projection
```

SwiGLU uses 3 weight matrices instead of 2 (standard FFN uses W1 and W2 only).
This costs 50% more parameters per layer but gives ~15% better quality per FLOP.

**Why SwiGLU?** The original transformer used ReLU: `relu(x @ W1) @ W2`. GPT-2/3
switched to GELU (smoother). Shazeer (2020, "GLU Variants Improve Transformer")
showed that adding a multiplicative gate — where one projection controls how much
of the other passes through — consistently improves quality across all model sizes.
The `d_ff` is reduced from `4d` to `8d/3` to compensate for the third matrix,
keeping total parameter count similar.

SwiGLU was adopted by PaLM (Google 2022), then Llama 1 (Meta 2023), and is now
universal. Every major LLM as of 2026 uses SwiGLU or the near-identical GeGLU
variant.

### Tied Embeddings

The output projection (hidden state → vocabulary logits) reuses the token embedding
matrix transposed:

```
logits = h @ token_emb.T    # (1024,) @ (1024, 32000) → (32000,)
```

This means the model doesn't learn separate input and output embeddings — one matrix
serves both purposes. It saves 32000 × 1024 × 2 = 65.5 MB of parameters and acts as
a regularizer.

**Why tie?** Press & Wolf (2017) showed that tying input and output embeddings
forces them to agree on token representations, which acts as regularization and
improves quality at small-to-medium model scales (<1B parameters). Larger models
like Llama 2/3 do NOT tie — at that scale, separate embeddings give more capacity.
At our 306M scale, tying is clearly beneficial.

---

## 6. Prefill and Decode

Text generation has two phases with fundamentally different performance characteristics.

### Prefill

**What:** Process the entire prompt at once, producing KV caches for all positions.

```
Input:  "The cat sat on" → [token_0, token_1, token_2, token_3]
Output: logits for the next token after each position, plus KV caches
```

Prefill processes all tokens in parallel — it's a batch matrix multiplication.
This is **compute-bound** because the batch dimension amortizes the weight loading:
each weight is loaded once but used for N tokens (where N = prompt length).

We use JAX (not Triton) for prefill because JAX's XLA compiler with cuDNN FlashAttention
is already highly optimized for parallel attention. Our prefill runs at ~814 tok/s
for a 128-token prompt (157 ms).

The implementation is in `model.py:195-232` (`prefill_with_kv()`):
```python
def prefill_with_kv(params, config, x):
    h = params["token_emb"][x]          # (seq_len, 1024) — embed all tokens
    k_caches, v_caches = [], []

    for layer in range(config["n_layers"]):
        h_norm = rms_norm(h, params[f"layer{layer}.ln1.scale"])

        # Project K and V, apply RoPE to K, store in caches
        k_proj = (h_norm @ params[f"layer{layer}.attn.k"]).reshape(seq_len, n_kv_heads, d_head)
        k_proj = apply_rope(k_proj, cos, sin)
        k_cache = zeros((n_kv_heads, max_seq, d_head), bf16)
        k_cache = k_cache.at[:, :seq_len, :].set(k_proj.transpose(1, 0, 2))

        # Run attention and FFN normally
        attn_out = causal_attention(h_norm, wq, wk, wv, wo, ...)
        h = h + attn_out
        h = h + swiGLU(rms_norm(h, ln2_scale))

    logits = rms_norm(h, ln_final) @ token_emb.T
    return logits, k_caches, v_caches
```

### Decode

**What:** Generate one token at a time. Each step takes the previous token, looks up
its KV cache entries from all previous positions, and produces the next token.

```
Step 0: process "the"   → reads KV cache positions [0..3], produces token 4
Step 1: process token 4  → reads KV cache positions [0..4], produces token 5
Step 2: process token 5  → reads KV cache positions [0..5], produces token 6
...
```

Decode is **bandwidth-bound** because we only process one token at a time.
Every weight is loaded from memory, multiplied with a single 1024-element vector,
and the result is added to an accumulator. The math is trivial — it's all
matrix-vector products — but we still need to read all 607 MB of weights every step.

This is where our custom Triton kernels shine. By fusing all 24 layers into a single
kernel launch, we:
1. Eliminate 24 × (multiple kernel launches per layer) = hundreds of launch overheads
2. Keep intermediate values (`h`) in registers across layers — never writing them
   to global memory and reading them back
3. Apply all optimizations (RoPE, RMSNorm, SwiGLU) inline without extra memory passes

---

## 7. Weight Packing

Before decode can run, we pack all model weights into a single flat bf16 buffer.
This eliminates per-layer pointer arithmetic and lets the kernel index weights
with simple offsets.

### `pack_weights()` — `fused_decode_nlayer.py:250-270`

```python
def pack_weights(params, config):
    """Pack per-layer weights into a single bf16 buffer.

    Layout per layer: ln1_s, wq, wk, wv, wo, ln2_s, gate, up, down
    """
    n_layers = config["n_layers"]
    layer_tensors = []
    for i in range(n_layers):
        p = f"layer{i}"
        layer_tensors.extend([
            params[f"{p}.ln1.scale"].reshape(-1),     # (1024,)
            params[f"{p}.attn.q"].reshape(-1),         # (1024, 1024) → (1048576,)
            params[f"{p}.attn.k"].reshape(-1),         # (1024, 256)  → (262144,)
            params[f"{p}.attn.v"].reshape(-1),         # (1024, 256)  → (262144,)
            params[f"{p}.attn.o"].reshape(-1),         # (1024, 1024) → (1048576,)
            params[f"{p}.ln2.scale"].reshape(-1),     # (1024,)
            params[f"{p}.ffn.gate"].reshape(-1),       # (1024, 2816) → (2883584,)
            params[f"{p}.ffn.up"].reshape(-1),         # (1024, 2816) → (2883584,)
            params[f"{p}.ffn.down"].reshape(-1),       # (2816, 1024) → (2883584,)
        ])
    return jnp.concatenate(layer_tensors).astype(jnp.bfloat16)
```

Each layer contributes:
```
ln1_scale:   1,024 elements
Wq:          1,024 × 1,024 = 1,048,576
Wk:          1,024 × 256   =   262,144      (D_KV = n_kv_heads × d_head = 4 × 64 = 256)
Wv:          1,024 × 256   =   262,144
Wo:          1,024 × 1,024 = 1,048,576
ln2_scale:   1,024
W_gate:      1,024 × 2,816 = 2,883,584
W_up:        1,024 × 2,816 = 2,883,584
W_down:      2,816 × 1,024 = 2,883,584
─────────────────────────────────────
Total per layer:          11,274,240 elements
× 24 layers:             270,581,760 elements
× 2 bytes (bf16):        541,163,520 bytes ≈ 541 MB
```

The kernel computes each layer's offset using `LAYER_W_SIZE`:
```python
LAYER_W_SIZE = (D_MODEL +                           # ln1_scale
                D_MODEL*D_MODEL + D_MODEL*D_KV +    # Wq + Wk
                D_MODEL*D_KV + D_MODEL*D_MODEL +    # Wv + Wo
                D_MODEL +                            # ln2_scale
                D_MODEL*D_FF + D_MODEL*D_FF +        # gate + up
                D_FF*D_MODEL)                        # down
```

Within each layer, individual weight matrices are located by accumulating offsets:
```python
off = w_base                          # start of this layer's weights
ln1_s_off = off;    off += D_MODEL    # first comes ln1_scale (1024 elements)
wq_off = off;       off += D_MODEL * D_MODEL  # then Wq (1024×1024 elements)
wk_off = off;       off += D_MODEL * D_KV     # then Wk (1024×256 elements)
...
```

### `prepare_decode_weights_nlayer()` — `fused_decode_nlayer.py:299-320`

This function prepares everything the decode kernel needs:

```python
def prepare_decode_weights_nlayer(params, config, vocab_size, kv_splits=2):
    n_heads = config["n_heads"]
    d_head = config["d_head"]
    output_vtile = 32
    total_blocks = n_heads * kv_splits
    align = output_vtile * total_blocks
    vocab_pad = ((vocab_size + align - 1) // align) * align
    pad_v = vocab_pad - vocab_size
```

**Vocab padding:** The output projection tiles over the vocabulary in chunks of
`OUTPUT_VTILE=32`. In the multi-SM kernel, these tiles are distributed across
`total_blocks` (= 16 heads × 1 kv_split = 16 blocks). For clean distribution,
vocab_size must be padded to a multiple of `output_vtile × total_blocks = 32 × 16 = 512`.

```
vocab_size = 32000
align = 512
vocab_pad = ceil(32000 / 512) * 512 = 63 * 512 = 32256
pad_v = 32256 - 32000 = 256 extra columns (padded with zeros)
```

The function returns a dictionary with:
```python
return {
    "token_emb": params["token_emb"].astype(jnp.bfloat16),     # (32000, 1024) bf16
    "packed_w": pack_weights(params, config),                    # flat bf16 buffer
    "lnf_s": params["ln_final.scale"].astype(jnp.bfloat16),    # (1024,) bf16
    "cos": cos.astype(jnp.bfloat16),                            # (512, 32) bf16
    "sin": sin.astype(jnp.bfloat16),                            # (512, 32) bf16
    "output_proj_padded": pad(emb_T, [(0,0),(0,pad_v)]).bf16,   # (1024, 32256) bf16
    "vocab_pad": vocab_pad,                                      # 32256
}
```

The output projection is the token embedding matrix transposed and padded:
`emb_T = params["token_emb"].T` gives shape (1024, 32000), then padded to (1024, 32256).

---

## 8. KV Cache Packing

Like weights, KV caches are packed into a single flat buffer so the kernel can
index them with simple arithmetic.

### `pack_kv_caches()` — `fused_decode_nlayer.py:273-283`

```python
def pack_kv_caches(k_caches, v_caches):
    """Pack per-layer KV caches into a single bf16 buffer.

    Input: k_caches[i] shape (n_kv_heads, max_seq, d_head) bf16
    Output: flat buffer with layout [layer0_k, layer0_v, layer1_k, layer1_v, ...]
    """
    parts = []
    for k, v in zip(k_caches, v_caches):
        parts.append(k.reshape(-1))
        parts.append(v.reshape(-1))
    return jnp.concatenate(parts)
```

Each layer's K and V caches are interleaved:

```
Layout: [L0_K | L0_V | L1_K | L1_V | ... | L23_K | L23_V]

Per-layer cache size:
  K cache: n_kv_heads × max_seq × d_head = 4 × 512 × 64 = 131,072 elements
  V cache: same = 131,072
  Total per layer: 262,144 elements = LAYER_KV_SIZE

  24 layers total: 6,291,456 elements × 2 bytes = 12.6 MB
```

The kernel locates each layer's caches using:
```python
LAYER_KV_SIZE = 2 * N_KV_HEADS * MAX_SEQ * D_HEAD    # 262,144
kv_base = layer * LAYER_KV_SIZE
kc_base = kv_base                                      # K cache starts at kv_base
vc_base = kv_base + N_KV_HEADS * MAX_SEQ * D_HEAD     # V cache starts after K
```

Within each cache, a specific position for a specific KV head is at:
```python
cache_off = kv_head * MAX_SEQ * D_HEAD    # offset to this head's data
# K[kv_head, pos, :] is at: kc_base + cache_off + pos * D_HEAD + dh
```

### `unpack_kv_caches()` — `fused_decode_nlayer.py:286-296`

The inverse operation, used after generation to extract per-layer caches:
```python
def unpack_kv_caches(packed, n_layers, n_kv_heads, max_seq, d_head):
    layer_kv_size = 2 * n_kv_heads * max_seq * d_head
    cache_size = n_kv_heads * max_seq * d_head
    k_caches, v_caches = [], []
    for i in range(n_layers):
        base = i * layer_kv_size
        k_caches.append(packed[base:base + cache_size].reshape(n_kv_heads, max_seq, d_head))
        v_caches.append(packed[base + cache_size:base + layer_kv_size].reshape(...))
    return k_caches, v_caches
```

---

## 9. The Single-SM Decode Kernel

`fused_decode_nlayer.py` contains a simpler decode kernel that runs on a **single SM**
(one block, `grid=(1,)`). It's slower than the multi-SM version but easier to
understand. We'll use it to build intuition before tackling the parallel version.

### Kernel Signature — `fused_decode_nlayer.py:37-67`

```python
@triton.jit
def _fused_decode_nlayer(
    token_emb_ptr,     # token embeddings (32000, 1024) bf16
    packed_w_ptr,      # all layer weights concatenated, bf16
    lnf_s_ptr,         # final RMSNorm scale (1024,) bf16
    output_proj_ptr,   # tied output projection (1024, vocab_pad) bf16
    cos_ptr, sin_ptr,  # RoPE tables (512, 32) bf16
    token_id_ptr,      # scalar: which token we're decoding
    pos_ptr,           # scalar: position in the sequence
    kv_in_ptr,         # packed KV caches (input from previous step)
    logits_ptr,        # OUTPUT: vocabulary logits (vocab_pad,) f32
    kv_out_ptr,        # OUTPUT: updated KV caches
    # Config (all compile-time constants)
    D_MODEL: tl.constexpr,    # 1024
    D_HEAD: tl.constexpr,     # 64
    D_FF: tl.constexpr,       # 2816
    N_HEADS: tl.constexpr,    # 16
    N_KV_HEADS: tl.constexpr, # 4
    D_KV: tl.constexpr,       # 256 (n_kv_heads × d_head = 4 × 64)
    N_LAYERS: tl.constexpr,   # 24
    MAX_SEQ: tl.constexpr,    # 512
    VOCAB_SIZE: tl.constexpr, # 32000
    VOCAB_PAD: tl.constexpr,  # 32256 (padded for tiling)
):
```

All config parameters are `tl.constexpr` — the compiler substitutes their actual
values and optimizes the resulting code. This means we get a different compiled
kernel for each model configuration, but each one is maximally optimized.

### Constants and Setup — Lines 29-34, 68-91

```python
BLOCK_K    = tl.constexpr(32)    # FFN tiling: process 32 hidden units at a time
VOCAB_TILE = tl.constexpr(128)   # output tiling: compute 128 logits at a time
KV_TILE    = tl.constexpr(64)    # attention tiling: process 64 KV positions at a time
OUTPUT_VTILE = tl.constexpr(32)  # output vocab tiling for multi-SM
```

Inside the kernel:
```python
d = tl.arange(0, D_MODEL)     # [0, 1, 2, ..., 1023] — index vector for d_model dimension
token_id = tl.load(token_id_ptr)  # which token to decode (scalar)
pos = tl.load(pos_ptr)            # position in sequence (scalar)
```

The `d` vector is the workhorse index — we use it to load entire 1024-dimensional
vectors in one instruction: `tl.load(ptr + d)` loads 1024 consecutive values.

### Embedding Lookup — Line 73

```python
h = tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
```

This loads the embedding for `token_id` from the embedding table. The embedding table
has shape (32000, 1024), stored as a flat array. To get row `token_id`, we jump to
offset `token_id * 1024` and load 1024 values.

Example: if token_id = 42, we load elements at indices [42×1024, 42×1024+1, ..., 42×1024+1023].

The `.to(tl.float32)` converts from bf16 (storage format) to f32 (computation format).
The hidden state `h` lives in f32 registers throughout all 24 layers — it's never
written back to global memory until the very end.

### Layer Loop — Line 93

```python
for layer in tl.range(N_LAYERS):  # 0, 1, 2, ..., 23
```

This is a dynamic loop — the body is compiled once and executed 24 times. Using
`tl.static_range` would unroll all 24 iterations, causing massive register spill
(the compiler tries to keep all 24 iterations' variables live simultaneously)
and 10+ minute compilation times.

### Weight and KV Cache Offsets — Lines 94-109

```python
    w_base = layer * LAYER_W_SIZE      # start of this layer's weights in packed_w
    kv_base = layer * LAYER_KV_SIZE    # start of this layer's KV caches
    kc_base = kv_base                  # K cache is first
    vc_base = kv_base + N_KV_HEADS * MAX_SEQ * D_HEAD  # V cache follows K

    # Individual weight offsets within this layer
    off = w_base
    ln1_s_off = off;    off += D_MODEL              # RMSNorm1 scale
    wq_off = off;       off += D_MODEL * D_MODEL    # query projection
    wk_off = off;       off += D_MODEL * D_KV       # key projection
    wv_off = off;       off += D_MODEL * D_KV       # value projection
    wo_off = off;       off += D_MODEL * D_MODEL    # output projection
    ln2_s_off = off;    off += D_MODEL              # RMSNorm2 scale
    gate_off = off;     off += D_MODEL * D_FF       # SwiGLU gate
    up_off = off;       off += D_MODEL * D_FF       # SwiGLU up
    down_off = off                                  # SwiGLU down
```

This is pure offset arithmetic — no memory accesses. Since LAYER_W_SIZE is a constexpr,
the compiler can compute these offsets at compile time.

### RMSNorm 1 — Lines 112-113

```python
    ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
    h_norm = ln_s * h * tl.math.rsqrt(tl.sum(h * h) / D_MODEL + 1e-5)
```

Step by step:
1. Load the learned scale parameter (1024 values)
2. Compute `h * h` — element-wise square of the hidden state
3. `tl.sum(h * h)` — sum all 1024 squared values (a reduction across all threads)
4. `/ D_MODEL` — divide by 1024 to get the mean
5. `+ 1e-5` — add epsilon to prevent division by zero
6. `tl.math.rsqrt(...)` — reciprocal square root (1/√x), a single fast instruction
7. `h * rsqrt(...)` — normalize h
8. `ln_s * ...` — multiply by learned scale

Note: this operates on the **single-SM kernel** where D_MODEL is a power of 2 (1024),
so no masking is needed. The multi-SM kernel (section 10) needs D_BLOCK padding and masks.

### Attention with RoPE and GQA — Lines 115-208

The attention computation loops over all 16 query heads:

```python
    attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
    h_norm_2d = h_norm[None, :].to(tl.bfloat16)  # (1, 1024) for tl.dot

    cos_val = tl.load(cos_ptr + pos * D_HALF + dh_lo).to(tl.float32)  # (32,)
    sin_val = tl.load(sin_ptr + pos * D_HALF + dh_lo).to(tl.float32)  # (32,)

    for head in tl.range(N_HEADS):  # 0..15
        kv_head = head // GQA_GROUP  # maps to 0..3
```

**RoPE cos/sin loading:** Position `pos` has 32 cos and 32 sin values (one per dimension
pair in d_head=64). These are loaded once before the head loop since they're the same
for all heads.

**GQA mapping:** `kv_head = head // GQA_GROUP` where GQA_GROUP = 16//4 = 4.
Heads 0-3 use KV head 0, heads 4-7 use KV head 1, etc.

#### Q/K Projections with RoPE — Lines 126-149

The Q and K projections are done in **two halves** for efficient RoPE application:

```python
        # Index vectors for this head's Q dimensions
        hd_lo = head * D_HEAD + dh_lo    # e.g., head=3: [192, 193, ..., 223]
        hd_hi = head * D_HEAD + dh_hi    # e.g., head=3: [224, 225, ..., 255]

        # Q projection — low half
        wq_lo = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd_lo[None, :])
        q_lo = tl.dot(h_norm_2d, wq_lo).to(tl.float32).sum(axis=0)  # (1,1024)@(1024,32)→(32,)

        # Q projection — high half
        wq_hi = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd_hi[None, :])
        q_hi = tl.dot(h_norm_2d, wq_hi).to(tl.float32).sum(axis=0)  # (32,)

        # RoPE on Q
        Q_lo = q_lo * cos_val - q_hi * sin_val
        Q_hi = q_lo * sin_val + q_hi * cos_val
```

Why split into halves? RoPE rotates pairs of dimensions: (q_0, q_32), (q_1, q_33), etc.
By projecting the low half (dims 0-31) and high half (dims 32-63) separately, we can
apply RoPE as simple multiply-and-add operations without any dimension shuffling.

The same pattern applies to K projection:
```python
        # K projection and RoPE (using KV head indices, not query head indices)
        wk_lo = tl.load(packed_w_ptr + wk_off + d[:, None] * D_KV + kv_hd_lo[None, :])
        k_lo = tl.dot(h_norm_2d, wk_lo).to(tl.float32).sum(axis=0)
        # ... similar for k_hi ...
        K_new_lo = k_lo * cos_val - k_hi * sin_val
        K_new_hi = k_lo * sin_val + k_hi * cos_val
```

Note that K uses `D_KV` (256) for column indexing, not `D_MODEL` (1024), because there
are only 4 KV heads × 64 dims = 256 K columns total.

#### V Projection — Lines 152-154

```python
        # V projection (no RoPE — only Q and K get rotary embeddings)
        kv_hd = kv_head * D_HEAD + dh  # full 64-dim index for this KV head
        wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_KV + kv_hd[None, :])
        V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)  # (64,)
```

V doesn't need the lo/hi split because it doesn't get RoPE.

#### KV Cache Write — Lines 157-159

```python
        # Store K (with RoPE) and V to output cache
        tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh_lo, K_new_lo.to(tl.bfloat16))
        tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh_hi, K_new_hi.to(tl.bfloat16))
        tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))
```

The K values are stored **after RoPE** — this is important. When we later read K from
the cache during attention, the rotation is already baked in. We don't need to re-apply
RoPE to cached K values.

#### Online Softmax (FlashAttention-style) — Lines 161-200

This is the most mathematically interesting part. Standard attention computes:

$$\text{attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d\_{\text{head}}}}\right) V$$

The naive implementation would:
1. Compute all scores QK^T — a vector of length `pos+1` (all previous positions)
2. Find the max, subtract it, exponentiate — the full softmax
3. Multiply softmax weights by V values

This requires storing all scores simultaneously. With 512 positions, that's fine (512
floats). But the pattern doesn't scale, and more importantly, it requires loading all
KV cache data at once.

**Why online softmax?** This is the same algorithmic idea behind FlashAttention
(Dao et al. 2022), which revolutionized transformer training and inference. The key
insight: softmax can be computed incrementally, processing one tile of K/V at a time,
by maintaining a running correction factor. This reduces memory from $O(n)$ to
$O(1)$ (we never store all scores) and improves cache locality (we process one small
tile, use it, discard it). In training, FlashAttention gives 2-4x speedup. In our
decode kernel, it lets us process the KV cache in 64-position tiles without ever
materializing the full attention vector.

**Online softmax** processes the KV cache in tiles of KV_TILE=64 positions, maintaining
a running estimate of the softmax that's corrected as new tiles arrive:

```python
        m_i = tl.full((1,), value=-1e9, dtype=tl.float32)  # running max score
        l_i = tl.zeros((1,), dtype=tl.float32)              # running sum of exp(scores)
        o_i = tl.zeros((D_HEAD,), dtype=tl.float32)          # running weighted sum of V

        for t in tl.range(0, MAX_SEQ, KV_TILE):  # 0, 64, 128, ..., 448
            tile_pos = t + tl.arange(0, KV_TILE)   # [t, t+1, ..., t+63]
            tile_mask = tile_pos <= pos              # only valid positions
```

For each tile of 64 KV positions:

**1. Load K tile and compute attention scores:**
```python
            K_tile_lo = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh_lo[None, :],
                               mask=tile_mask[:, None], other=0.0)
            # ... similarly for K_tile_hi ...

            # Overlay the new K we just computed (it's not in kv_in yet)
            K_tile_lo = tl.where(tile_pos[:, None] == pos, K_new_lo[None, :], K_tile_lo)

            # Copy to output KV cache
            tl.store(kv_out_ptr + kc_base + ..., K_tile_lo, mask=tile_mask[:, None])

            # Score = Q · K (dot product split across lo/hi halves)
            s = (tl.sum(Q_lo[None, :] * K_tile_lo, axis=1)
               + tl.sum(Q_hi[None, :] * K_tile_hi, axis=1)) * scale
            s = tl.where(tile_mask, s, -1e9)  # mask future positions
```

The score is `Q · K / sqrt(d_head)` where `scale = 1 / sqrt(64) ≈ 0.125`.

**2. Online softmax update:**
```python
            m_ij = tl.max(s)                    # max score in this tile
            m_new = tl.maximum(m_i, m_ij)       # new global max
            alpha = tl.exp(m_i - m_new)         # correction factor for previous tiles
            p = tl.exp(s - m_new)               # softmax weights for this tile
            l_i = l_i * alpha + tl.sum(p)       # update running sum
            o_i = o_i * alpha + tl.sum(p[:, None] * V_tile, axis=0)  # update weighted sum
            m_i = m_new
```

Here's the key insight: when we find a new maximum in tile j that's larger than our
running maximum, all previous softmax weights need to be scaled down. Instead of
going back and re-scaling them, we apply a correction factor `alpha = exp(m_old - m_new)`.

Numerical example with 3 tiles of 2 values each:

```
Tile 0: scores = [3.0, 1.0], V = [[1,0], [0,1]]
  m_i = -1e9 → m_new = 3.0
  alpha = exp(-1e9 - 3.0) ≈ 0 (initial, doesn't matter)
  p = [exp(3-3), exp(1-3)] = [1.0, 0.135]
  l_i = 0 * 0 + 1.135 = 1.135
  o_i = [0,0] * 0 + [1.0*[1,0] + 0.135*[0,1]] = [1.0, 0.135]

Tile 1: scores = [5.0, 2.0], V = [[1,1], [0,0]]
  m_i = 3.0 → m_new = 5.0
  alpha = exp(3.0 - 5.0) = 0.135 (scale down previous tiles!)
  p = [exp(5-5), exp(2-5)] = [1.0, 0.050]
  l_i = 1.135 * 0.135 + 1.050 = 1.203
  o_i = [1.0, 0.135] * 0.135 + [1.0*[1,1] + 0.050*[0,0]] = [1.135, 1.018]

Final: attn_out = o_i / l_i = [0.943, 0.846]
```

This gives the exact same result as computing the full softmax over all 4 scores
[3, 1, 5, 2], but we never stored more than 2 scores at a time.

**3. Final attention output:**
```python
        attn_out = o_i / l_i  # (D_HEAD,) — normalize by total softmax weight
```

#### O Projection — Lines 204-207

```python
        hd = head * D_HEAD + dh
        wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :])
        attn_accum += tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)
```

Each head produces a (D_HEAD,)-dimensional output. The O projection maps it back
to D_MODEL dimensions. All 16 heads' O projections are summed in `attn_accum`.

After the head loop, the attention residual is added:
```python
    h = h + attn_accum  # (D_MODEL,) — residual connection
```

### SwiGLU FFN — Lines 211-230

```python
    # RMSNorm 2
    ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
    h_norm = ln_s * h * tl.math.rsqrt(tl.sum(h * h) / D_MODEL + 1e-5)
    h_norm_2d = h_norm[None, :].to(tl.bfloat16)  # (1, 1024) for tl.dot

    ffn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):  # 0, 32, 64, ..., 2784
        kk = k + tl.arange(0, BLOCK_K)    # [k, k+1, ..., k+31]
```

The FFN is tiled over the hidden dimension (d_ff=2816) in chunks of BLOCK_K=32.
Each tile:

```python
        # Gate projection: (1, 1024) @ (1024, 32) → (32,)
        gate_w = tl.load(packed_w_ptr + gate_off + d[:, None] * D_FF + kk[None, :])
        gate = tl.dot(h_norm_2d, gate_w).to(tl.float32).sum(axis=0)

        # Up projection: (1, 1024) @ (1024, 32) → (32,)
        up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :])
        up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)

        # SwiGLU activation: SiLU(gate) * up
        act = (gate * tl.sigmoid(gate)) * up

        # Down projection: (1, 32) @ (32, 1024) → (1024,)
        down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :])
        ffn_accum += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)
```

The tiling pattern: we compute 32 hidden units of the FFN at a time. For each tile,
we load the gate and up weights, compute the gated activation, then immediately
project back down and accumulate. This way, we never materialize the full 2816-dimensional
hidden state — only 32 elements at a time, fitting comfortably in registers.

After the FFN loop:
```python
    h = h + ffn_accum  # residual connection
```

### Output: Final RMSNorm + Tied Projection — Lines 232-245

After all 24 layers, one final RMSNorm and output projection:

```python
    # Final RMSNorm
    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    h_final = ln_s * h * tl.math.rsqrt(tl.sum(h * h) / D_MODEL + 1e-5)
    h_final_2d = h_final[None, :].to(tl.bfloat16)

    # Tied output projection: h @ token_emb.T, tiled over vocabulary
    for v_start in tl.range(0, VOCAB_PAD, OUTPUT_VTILE):  # 0, 32, 64, ..., 32224
        vv = v_start + tl.arange(0, OUTPUT_VTILE)          # [v_start, ..., v_start+31]
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :])
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)  # mask padding
        tl.store(logits_ptr + vv, tile_logits)
```

The output projection computes 32 logits at a time: (1, 1024) @ (1024, 32) → (32,).
Padded vocab entries beyond VOCAB_SIZE get -1e9 (negative infinity), so they can never
be selected.

### Python Wrapper — `fused_decode_nlayer()` at line 323

```python
def fused_decode_nlayer(w, config, token_id, pos, kv_packed, vocab_size):
    ...
    logits_pad, kv_out = jt.triton_call(
        w["token_emb"], w["packed_w"], w["lnf_s"], w["output_proj_padded"],
        w["cos"], w["sin"],
        jnp.int32(token_id), jnp.int32(pos),
        kv_packed,
        kernel=_fused_decode_nlayer,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),    # logits
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),  # updated KV caches
        ],
        grid=(1,),          # single block = single SM
        num_warps=4,        # 128 threads per block
        num_stages=1,       # no double-buffering (not enough shared memory)
        ...
    )
    return logits_pad[:vocab_size], kv_out
```

`jt.triton_call` is the bridge between JAX and Triton: it takes JAX arrays, passes
them to the Triton kernel as raw pointers, and wraps the outputs back as JAX arrays.
`out_shape` tells JAX what output arrays to allocate.

`grid=(1,)` means one block, one SM. This is the simplest launch — no parallelism,
no synchronization needed. The downside: 79 of our 80 SMs sit idle.

---

## 10. The Multi-SM Decode Kernel

The multi-SM kernel (`kernels/multi_sm_decode.py`) is our fast decode kernel. It does
the same computation as the single-SM kernel but distributes work across all 16 blocks
(one per attention head, with `kv_splits=1`), using the entire GPU.

### Why Multi-SM?

The single-SM kernel runs on one SM. The RTX 4080 Super has 80 SMs. Even though
inference is bandwidth-bound (not compute-bound), using multiple SMs helps because:

1. **Multiple memory controllers:** Different SMs can issue memory requests in parallel,
   better saturating the HBM bandwidth channels
2. **Parallelizable attention:** Each head's attention is independent — 16 heads can
   run on 16 SMs simultaneously
3. **Distributed FFN:** The 2816-wide FFN can be split across blocks, each handling
   a portion

But multi-SM introduces a problem: blocks need to **synchronize** at certain points.
For example, all blocks must finish attention before any block starts the FFN (because
the FFN input depends on attention output). This requires barriers.

### Architecture Overview — `multi_sm_decode.py:1-21`

```
grid = (N_HEADS * KV_SPLITS,) = (16 * 1,) = 16 blocks

Per layer, 3 phases with barriers between them:

Phase 1: [all 16 blocks independently]
  RMSNorm1 → QKV projection → RoPE → attention → O projection
  Each block handles ONE attention head
  → write results to workspace → BARRIER

Phase 2: [all 16 blocks independently]
  Merge attention from all heads → residual → RMSNorm2 → SwiGLU FFN
  Each block handles D_FF/16 ≈ 176 hidden units of FFN
  → write FFN partial to workspace → BARRIER

Phase 3: [all 16 blocks independently]
  Sum FFN partials from all blocks → residual → h ready for next layer
  → BARRIER (prevent next layer from overwriting before all reads complete)
```

### D_BLOCK Padding — Lines 72-75

```python
    pid = tl.program_id(0)               # 0..15, one per head
    head_id = pid // KV_SPLITS           # with KV_SPLITS=1: head_id = pid
    kv_split = pid % KV_SPLITS           # with KV_SPLITS=1: always 0
    d = tl.arange(0, D_BLOCK)            # [0, 1, ..., D_BLOCK-1]
    d_mask = d < D_MODEL                 # [True, True, ..., True] when D_BLOCK == D_MODEL
```

`tl.arange` requires a **power-of-2** argument. D_MODEL=1024 happens to already be a
power of 2, so D_BLOCK=1024 and d_mask is all True. But the code handles the general
case: if D_MODEL were 768, D_BLOCK would be 1024, and d_mask would be True for indices
0-767 and False for 768-1023.

The mask is applied everywhere data touches D_MODEL dimensions:
```python
h = tl.load(ptr + d, mask=d_mask, other=0.0)   # load with padding
tl.store(ptr + d, value, mask=d_mask)           # store only valid elements
h_sq = tl.where(d_mask, h * h, 0.0)            # zero padding in RMSNorm variance
```

This is computed by `_next_power_of_2()` in the Python wrapper:
```python
def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p
```

### Workspace Layout — Lines 80-85, 389-397

The kernel uses a shared workspace buffer for cross-block communication. It has
separate regions to avoid **race conditions** (explained below):

```python
    attn_partial_ptr = workspace_ptr                     # attention O-proj results
    ffn_partial_ptr = workspace_ptr + FFN_PARTIAL_OFF    # FFN partial results (separate!)
    attn_ml_ptr = workspace_ptr + ATTN_ML_OFF            # attention m and l values
    barrier_ptr = workspace_ptr + BARRIER_OFF            # barrier counters
    done_ptr = workspace_ptr + DONE_OFF                  # done flags (unused but allocated)
    argmax_ptr = workspace_ptr + ARGMAX_OFF              # per-block argmax results
```

The Python wrapper computes these offsets:
```python
    # ffn_partial_off = total_blocks * d_block
    #   → attn partials: [0, ffn_partial_off)  — 16 × 1024 = 16,384 floats
    # attn_ml_off = ffn_partial_off + total_blocks * d_block
    #   → FFN partials: [ffn_partial_off, attn_ml_off) — 16 × 1024 = 16,384 floats
    # barrier_off = attn_ml_off + total_blocks * 2
    #   → m/l pairs: total_blocks × 2 = 32 floats
    # done_off = barrier_off + n_barriers
    #   → n_barriers = 3 × 24 + 1 = 73 barrier counters
    # argmax_off = done_off + n_barriers
    #   → argmax: total_blocks × 2 = 32 floats (value + index per block)
```

**Why separate attn and FFN buffers?** In earlier versions, both attention O-projection
results and FFN partial sums shared the same buffer region. This caused a race condition:
when a fast block finished Phase 1 (writing attention output) and moved to Phase 2
(writing FFN partials), it overwrote data that a slow block hadn't read yet. Separating
the buffers eliminated this bug.

### Projection Tiling (PROJ_TILE) — Lines 139-152

The single-SM kernel loads full weight matrices like `(D_MODEL, D_HEAD) = (1024, 64)`.
This is 1024 × 64 × 2 = 128 KB in bf16, which exceeds the 101 KB shared memory limit.

The multi-SM kernel tiles projections with PROJ_TILE=512:

```python
    PROJ_TILE = tl.constexpr(512)

    Q = tl.zeros((D_HEAD,), dtype=tl.float32)
    K_new = tl.zeros((D_HEAD,), dtype=tl.float32)
    V_new = tl.zeros((D_HEAD,), dtype=tl.float32)

    for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):  # dd = 0, 512
        dt = dd + tl.arange(0, PROJ_TILE)   # [0..511], then [512..1023]
        dt_mask = dt < D_MODEL

        # Load h_norm tile from scratch buffer
        h_tile = tl.load(attn_partial_ptr + pid * D_BLOCK + dt,
                         mask=dt_mask, other=0.0).to(tl.bfloat16)

        # Q projection tile: (1, 512) @ (512, 64) → (64,)
        wq_t = tl.load(packed_w_ptr + wq_off + dt[:, None] * D_MODEL + hd[None, :],
                       mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
        Q += tl.dot(h_tile[None, :], wq_t).to(tl.float32).sum(axis=0)

        # K, V projections similarly...
```

Two iterations: first tile processes h[0:512] × W[0:512, :], second processes
h[512:1024] × W[512:1024, :]. Each tile loads (512, 64) = 64 KB of weights, well
within shared memory limits.

The `.sum(axis=0)` after `tl.dot` is necessary because tl.dot returns a 2D result
even for (1, 512) × (512, 64) — it gives (1, 64), and `.sum(axis=0)` collapses the
first dimension.

Note `h_norm` is **stored to the workspace buffer** before the projection loop:
```python
    tl.store(attn_partial_ptr + pid * D_BLOCK + d, h_norm, mask=d_mask)
```
This is because the tiled projection reads `h_norm` in chunks, and we can't read
different subsets of a register vector — we need to write it to accessible memory first.

### RoPE via Scratch Buffer — Lines 154-173

The multi-SM kernel applies RoPE through a scratch buffer because RoPE shuffles
dimensions (it needs to read q_lo and q_hi separately, which requires the values
to be in addressable memory, not just registers):

```python
    cos_val = tl.load(cos_ptr + pos * D_HALF + rope_lo).to(tl.float32)
    sin_val = tl.load(sin_ptr + pos * D_HALF + rope_lo).to(tl.float32)

    scratch = attn_partial_ptr + pid * D_BLOCK  # reuse workspace

    # Store Q to scratch, read halves, rotate, read back
    tl.store(scratch + dh, Q)
    q_lo = tl.load(scratch + rope_lo)           # first 32 elements
    q_hi = tl.load(scratch + D_HALF + rope_lo)  # last 32 elements
    tl.store(scratch + rope_lo, q_lo * cos_val - q_hi * sin_val)
    tl.store(scratch + D_HALF + rope_lo, q_lo * sin_val + q_hi * cos_val)
    Q = tl.load(scratch + dh)                   # read rotated Q back
```

This store-rotate-load pattern is necessary because Triton doesn't support arbitrary
indexing into register vectors — you can't write `Q[0:32]` to get the first half of
a register-resident vector. Writing to memory, reading specific indices, and writing
back is the workaround.

### KV Cache Writes — Lines 175-208

```python
    # Only one block per kv_head writes to avoid redundant concurrent stores
    is_kv_primary = (head_id == kv_head * GQA_GROUP)
    if is_kv_primary:
        tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))
```

With GQA, multiple query heads (e.g., heads 0-3) share the same KV head (KV head 0).
All 4 blocks compute the same K_new and V_new for KV head 0, but only one needs to
write it. `is_kv_primary` ensures only head 0 (the first in the group) writes.

During the attention loop, the same principle applies — only primary blocks write
back KV cache tiles:
```python
            K_tile = tl.where(tile_pos[:, None] == pos, K_new[None, :], K_tile)
            if is_kv_primary:
                tl.store(kv_out_ptr + ..., K_tile.to(tl.bfloat16), mask=tile_mask[:, None])
```

### L2 Cache Eviction Hints — Lines 194-204

```python
            K_tile = tl.load(kv_in_ptr + ...,
                            mask=tile_mask[:, None], other=0.0,
                            eviction_policy='evict_last')    # keep in L2
```

`eviction_policy='evict_last'` tells the L2 cache controller: "this data is likely
to be needed again, evict it last." KV cache data is reused across decode steps —
position 0's K values are read at every step — so keeping it in L2 avoids HBM
round-trips.

Later, for the output projection:
```python
        out_w = tl.load(output_proj_ptr + ...,
                        eviction_policy='evict_first')   # evict immediately
```

Output projection weights are used once and never again (they're not shared across
layers), so we hint `evict_first` to make room for more useful data.

These hints give ~3% throughput improvement.

### Attention Scores — Lines 210-218

```python
            s = tl.sum(Q[None, :] * K_tile, axis=1) * scale
            s = tl.where(tile_mask, s, -1e9)
```

Unlike the single-SM kernel (which splits Q into lo/hi halves), the multi-SM kernel
uses the full 64-dim Q and K_tile. The RoPE is already applied via the scratch buffer,
so Q has the complete rotated vector.

The online softmax update is identical to the single-SM version:
```python
            m_ij = tl.max(s)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p)
            o_i = o_i * alpha + tl.sum(p[:, None] * V_tile, axis=0)
            m_i = m_new
```

### O Projection (Tiled) — Lines 224-230

```python
    for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
        dt = dd + tl.arange(0, PROJ_TILE)
        dt_mask = dt < D_MODEL
        wo_t = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + dt[None, :],
                       mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
        o_tile = tl.dot(attn_out[None, :].to(tl.bfloat16), wo_t).to(tl.float32).sum(axis=0)
        tl.store(attn_partial_ptr + pid * D_BLOCK + dt, o_tile, mask=dt_mask)
```

Same PROJ_TILE=512 tiling as for QKV projections. The O projection result is stored
to the workspace for the next phase (merging attention outputs from all heads).

After the O projection, the block also stores its attention statistics (m and l):
```python
    tl.store(attn_ml_ptr + pid * 2, tl.sum(m_i))
    tl.store(attn_ml_ptr + pid * 2 + 1, tl.sum(l_i))
```

### Atomic Barriers and Memory Fences — Lines 234-241

This is the trickiest part of the multi-SM kernel. After Phase 1 (attention), all
blocks must wait for each other before Phase 2 can read their results.

```python
    # Global memory fence: ensure all stores are visible to other SMs
    tl.debug_barrier()
    tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;",
                              "=r", [], dtype=tl.int32, is_pure=False, pack=1)

    # Barrier: each block increments counter, then spins until all blocks arrive
    b0 = layer * 3          # barrier index for this phase of this layer
    tl.atomic_add(barrier_ptr + b0, 1.0, sem='acq_rel', scope='gpu')
    while tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
        pass

    # Another memory fence after passing the barrier
    tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;",
                              "=r", [], dtype=tl.int32, is_pure=False, pack=1)
```

Let's break this down:

**`tl.debug_barrier()`** — A block-level barrier that ensures all threads within this
block have completed their stores. This is a sync point within one block (128 threads).

**`membar.gl` (global memory barrier)** — This is inline PTX assembly (the lowest level
of GPU programming). `membar.gl` is a "memory barrier, global level" instruction that
ensures all previous memory writes from this SM are visible to all other SMs. Without
this, the Triton compiler might reorder stores to appear after the atomic barrier,
meaning other blocks would pass the barrier but read stale data.

The unusual syntax `"membar.gl; mov.u32 $0, 0;"` adds a dummy `mov` instruction because
`tl.inline_asm_elementwise` requires an output — the `membar.gl` itself doesn't produce
one. The `"=r"` constraint says "output to a register," `is_pure=False` prevents the
compiler from optimizing it away, and `pack=1` says it's a scalar operation.

**Atomic barrier pattern:**
```python
    # Step 1: "I've arrived" — atomically increment the counter
    tl.atomic_add(barrier_ptr + b0, 1.0, sem='acq_rel', scope='gpu')

    # Step 2: spin-wait until all blocks have arrived
    while tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
        pass
```

Each block atomically adds 1 to the barrier counter. Then it spins, reading the counter
(adding 0 is a read that goes through the atomic unit to get the latest value) until
it reaches TOTAL_BLOCKS (16). When all 16 blocks have incremented, the counter equals 16
and everyone proceeds.

`sem='acq_rel'` on the increment means:
- `acquire`: all memory reads after this see the latest writes from other threads
- `release`: all memory writes before this are visible to other threads

`sem='acquire'` on the read means: when we see the counter reach 16, we're guaranteed
to also see all the stores that other blocks made before their increments.

`scope='gpu'` means this applies across all SMs on the GPU (not just within one SM).

**Why 3 barriers per layer?** Each layer has:
- Barrier 0 (`layer*3`): after Phase 1 (attention) — before merging attention outputs
- Barrier 1 (`layer*3+1`): after Phase 2 (FFN) — before reducing FFN partials
- Barrier 2 (`layer*3+2`): after Phase 3 (reduce) — before next layer overwrites buffers

Plus one final barrier (`3*N_LAYERS`) for the output projection.

Total barrier count: `3 × 24 + 1 = 73`. These need separate counter slots in the
workspace buffer — the barrier slot overflow bug (where 73 barriers were allocated
only 32 slots, causing barriers to corrupt adjacent memory) was one of the hardest
bugs to find.

### Phase 2: Merge Attention + FFN — Lines 243-291

After the barrier, each block reads all other blocks' attention results and merges them:

```python
    # Merge attention outputs from all heads with online softmax correction
    attn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
    for head in tl.range(N_HEADS):    # 0..15
        m_max_val = tl.full((), -1e9, dtype=tl.float32)
        # Find max m across KV splits for this head
        for s in tl.static_range(KV_SPLITS):  # with KV_SPLITS=1, just one iteration
            m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
            m_max_val = tl.maximum(m_max_val, m_s)

        # Merge with correction weights
        l_total = tl.full((), 0.0, dtype=tl.float32)
        o_merged = tl.zeros((D_BLOCK,), dtype=tl.float32)
        for s in tl.static_range(KV_SPLITS):
            m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
            l_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2 + 1)
            o_s = tl.load(attn_partial_ptr + (head * KV_SPLITS + s) * D_BLOCK + d,
                          mask=d_mask, other=0.0)
            w = l_s * tl.exp(m_s - m_max_val)    # correction weight
            l_total = l_total + w
            o_merged = o_merged + o_s * w
        attn_total = attn_total + o_merged / l_total
    h = h + attn_total
```

With KV_SPLITS=1, there's only one "split" per head, so the merge is trivially
`attn_total += o_s / l_s` (the correction factor exp(m - m) = 1). The merge logic
exists for KV_SPLITS>1 where each head's attention is split across multiple blocks
for even more parallelism — but kv_splits=1 turned out to be more reliable.

#### Distributed SwiGLU FFN — Lines 270-291

```python
    # Each block handles a slice of the FFN hidden dimension
    ff_start = pid * FF_PER_BLOCK
    ffn_partial = tl.zeros((D_BLOCK,), dtype=tl.float32)

    for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):  # BLOCK_K = 16
        kk = ff_start + k + tl.arange(0, BLOCK_K)
        ff_mask = kk < D_FF
        # Gate: (1, D_BLOCK) @ (D_BLOCK, 16) → (16,)
        gate_w = tl.load(packed_w_ptr + gate_off + d[:, None] * D_FF + kk[None, :],
                         mask=d_mask[:, None] & ff_mask[None, :], other=0.0)
        gate = tl.dot(h_norm_2d, gate_w).to(tl.float32).sum(axis=0)
        # Up: similar
        up_w = tl.load(...)
        up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
        # SwiGLU
        act = (gate * tl.sigmoid(gate)) * up
        # Down: (1, 16) @ (16, D_BLOCK) → (D_BLOCK,)
        down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :],
                         mask=ff_mask[:, None] & d_mask[None, :], other=0.0)
        ffn_partial += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)
```

**FF_PER_BLOCK calculation** (Python wrapper, lines 385-387):
```python
    raw_ff = (d_ff + total_blocks - 1) // total_blocks   # ceil(2816 / 16) = 176
    ff_per_block = ((raw_ff + block_k - 1) // block_k) * block_k  # ceil(176/16)*16 = 176
```

Each block handles 176 hidden units (out of 2816 total). With BLOCK_K=16, that's
11 tiles per block. The `ff_mask = kk < D_FF` handles the last block cleanly when
D_FF isn't evenly divisible.

Note BLOCK_K=16 in the multi-SM kernel vs BLOCK_K=32 in the single-SM kernel. The
multi-SM kernel loads both gate and up weights simultaneously (both (D_BLOCK, BLOCK_K)
tiles), so each tile is 1024 × 16 × 2 = 32 KB. Two tiles = 64 KB, fitting within
shared memory. With BLOCK_K=32, each tile would be 64 KB, and two would be 128 KB >
101 KB.

The FFN partial is stored to a **separate buffer** from attention partials:
```python
    tl.store(ffn_partial_ptr + pid * D_BLOCK + d, ffn_partial, mask=d_mask)
```

### Phase 3: Reduce FFN + Residual — Lines 301-312

```python
    # Sum FFN partials from all blocks
    ffn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
    for i in tl.range(TOTAL_BLOCKS):  # 0..15
        ffn_total += tl.load(ffn_partial_ptr + i * D_BLOCK + d, mask=d_mask, other=0.0)
    h = h + ffn_total

    # Barrier to prevent next layer from overwriting before all reads complete
    tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", ...)
    b2 = layer * 3 + 2
    tl.atomic_add(barrier_ptr + b2, 1.0, sem='acq_rel', scope='gpu')
    while tl.atomic_add(barrier_ptr + b2, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
        pass
```

Every block reads all 16 FFN partials and sums them. This is **redundant computation**
— all 16 blocks compute the same sum. But it's faster than having one block compute
the sum and broadcasting it (which would require another barrier).

The final barrier ensures no block starts the next layer (which reuses the workspace
buffers) before all blocks have finished reading.

### Output: Final RMSNorm + Distributed Argmax — Lines 314-357

```python
    # Final RMSNorm (same as within layers)
    ln_s = tl.load(lnf_s_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    h_sq = tl.where(d_mask, h * h, 0.0)
    h_final = tl.where(d_mask,
                       ln_s * h * tl.math.rsqrt(tl.sum(h_sq) / D_MODEL + 1e-5),
                       0.0)
    h_final_2d = h_final[None, :].to(tl.bfloat16)
```

#### Distributed Output Projection

```python
    # Each block handles TILES_PER_BLOCK tiles of the vocabulary
    TILES_PER_BLOCK = VOCAB_PAD // (OUTPUT_VTILE * TOTAL_BLOCKS)
    #  = 32256 // (32 * 16) = 63 tiles per block

    best_val = -1e9
    best_idx = 0.0
    for tile_idx in tl.range(0, TILES_PER_BLOCK):
        v_start = (pid * TILES_PER_BLOCK + tile_idx) * OUTPUT_VTILE
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :],
                        mask=d_mask[:, None], other=0.0,
                        eviction_policy='evict_first').to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)

        # Track local best for in-kernel argmax
        tile_max = tl.max(tile_logits)
        if tile_max > best_val:
            best_val = tile_max
            best_idx = (v_start + tl.argmax(tile_logits, axis=0)).to(tl.float32)
```

The vocabulary is striped across blocks: block 0 handles vocab indices
[0..62×32-1], block 1 handles [63×32..125×32-1], etc. Each block computes 63 tiles
of 32 logits each = 2016 logits per block, totaling 16 × 2016 = 32,256 logits.

#### In-Kernel Argmax — Lines 339-357

Finding the best token entirely on the GPU avoids a round-trip to the CPU:

```python
    # Step 1: each block writes its local best value and index
    tl.store(argmax_ptr + pid * 2, best_val)
    tl.store(argmax_ptr + pid * 2 + 1, best_idx)

    # Barrier to ensure all blocks have written
    tl.inline_asm_elementwise("membar.gl; ...", ...)
    tl.atomic_add(barrier_ptr + N_BARRIERS, 1.0, sem='acq_rel', scope='gpu')
    while tl.atomic_add(barrier_ptr + N_BARRIERS, 0.0, ...) < TOTAL_BLOCKS:
        pass

    # Step 2: block 0 finds the global best
    if pid == 0:
        global_best_val = -1e9
        global_best_idx = 0.0
        for i in tl.range(TOTAL_BLOCKS):    # 0..15
            v = tl.load(argmax_ptr + i * 2)
            idx = tl.load(argmax_ptr + i * 2 + 1)
            if v > global_best_val:
                global_best_val = v
                global_best_idx = idx
        tl.store(next_token_ptr, global_best_idx.to(tl.int32))
```

Two-level argmax: each block finds its best among ~2016 logits, then block 0 finds
the global best among the 16 per-block winners. This avoids transferring all 32000
logits to the CPU for argmax — only the single winning token ID comes back.

The `next_token_ptr` output is a single int32 — the predicted next token. When using
greedy decoding (temperature=0), this is all generate.py needs from the kernel.

### Python Wrapper — `multi_sm_decode_nlayer()` at line 369

```python
def multi_sm_decode_nlayer(w, config, token_id, pos, kv_packed, vocab_size, kv_splits=2):
    ...
    logits_pad, kv_out, next_token = jt.triton_call(
        w["token_emb"], w["packed_w"], w["lnf_s"], w["output_proj_padded"],
        w["cos"], w["sin"],
        jnp.int32(token_id), jnp.int32(pos),
        kv_packed,
        workspace,
        kernel=_multi_sm_decode,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),       # logits
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),  # updated KV
            jax.ShapeDtypeStruct((1,), jnp.int32),                 # next_token
        ],
        grid=(total_blocks,),    # 16 blocks (one per head)
        num_warps=4,
        num_stages=1,
        ...
    )
    return next_token[0], logits_pad[:vocab_size], kv_out
```

The key differences from the single-SM wrapper:
1. `grid=(total_blocks,)` instead of `grid=(1,)` — 16 blocks
2. A `workspace` buffer is passed in for cross-block communication
3. Three outputs instead of two — `next_token` is the in-kernel argmax result
4. Extra constexpr parameters for workspace offsets, block work distribution, etc.

---

## 11. The Generation Loop

`generate.py` orchestrates the end-to-end text generation pipeline.

### Prefill — `_prefill()` at line 82

```python
def _prefill(params, config, prompt_ids, vocab_size):
    ctx_len = config["context_len"]      # 512
    prompt_len = len(prompt_ids)
    # Pad prompt to context length (JAX needs fixed shapes for XLA compilation)
    x = jnp.pad(prompt_ids, (0, ctx_len - prompt_len)).astype(jnp.int32)

    # Run JAX prefill: processes all tokens in parallel, returns logits + KV caches
    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()  # force synchronous execution
```

`block_until_ready()` forces JAX to complete the computation before continuing.
JAX normally uses lazy evaluation — computations are queued and executed later.
For benchmarking and sequencing, we need the result now.

```python
    # Prepare decode weights (pack all layer weights, prepare output proj, etc.)
    w = prepare_decode_weights_nlayer(params, config, vocab_size, kv_splits=1)

    # Pack KV caches into flat buffer for Triton kernel
    kv_packed = pack_kv_caches(k_caches, v_caches)

    # First decode token comes from the last prompt position's logits
    first_logits = logits[prompt_len - 1]
```

**kv_splits=1:** We use 1 KV split (not 2) because kv_splits=2 caused non-determinism
at 24 layers due to excessive barrier interaction noise. With kv_splits=1, grid=(16,)
with one block per head.

```python
    # Warmup decode step: triggers Triton JIT compilation
    warmup_tok = jnp.argmax(first_logits)
    _tok, _, _kv = multi_sm_decode_nlayer(
        w, config, warmup_tok, prompt_len, kv_packed, vocab_size, kv_splits=1)
    _ = int(_tok)  # force execution
```

The first call to a Triton kernel triggers JIT compilation (~1-3 seconds). We do this
during prefill so the actual generation loop doesn't include compilation time. The
warmup result is discarded — we restart decode from the first_logits.

### Streaming Generation — `stream_tokens()` at line 107

```python
def stream_tokens(params, config, prompt_ids, max_tokens=128,
                  temperature=0.0, top_p=1.0, rep_penalty=1.0, seed=None):
    """Yield token IDs one at a time with optional sampling."""
    if seed is not None:
        np.random.seed(seed)

    vocab_size = config["vocab_size"]
    w, first_logits, start_pos, kv_packed = _prefill(
        params, config, prompt_ids, vocab_size)

    generated = []

    # First token from prefill logits
    if temperature == 0.0 and rep_penalty == 1.0:
        first_id = int(jnp.argmax(first_logits))
    else:
        first_id = sample_token(first_logits, temperature, top_p,
                                rep_penalty, generated)
    generated.append(first_id)
    yield first_id
```

The first token uses prefill logits (the logits at the last prompt position). This is
yielded immediately — the caller sees tokens as they're generated.

```python
    tok = jnp.int32(first_id)
    for i in range(max_tokens - 1):
        # Run one decode step: produces next token + updated KV cache
        tok_out, logits, kv_packed = multi_sm_decode_nlayer(
            w, config, tok, start_pos + i, kv_packed, vocab_size, kv_splits=1)

        if temperature == 0.0 and rep_penalty == 1.0:
            token_id = int(tok_out)    # use in-kernel argmax (greedy)
        else:
            token_id = sample_token(logits, temperature, top_p,
                                    rep_penalty, generated)

        generated.append(token_id)
        yield token_id
        tok = jnp.int32(token_id)
```

Each iteration:
1. Calls the multi-SM kernel with the previous token and its position
2. Gets back: `tok_out` (greedy argmax from the kernel), `logits` (full vocabulary
   scores), and `kv_packed` (updated KV caches)
3. For greedy decoding: uses `tok_out` directly (no CPU-side logits processing)
4. For sampling: transfers `logits` to CPU and runs `sample_token()`
5. Yields the token ID and feeds it back for the next step

The `kv_packed` returned from each step becomes the input for the next step — it
contains all KV caches updated with the new position's K and V values.

### Sampling — `sample_token()` at line 28

```python
def sample_token(logits, temperature=1.0, top_p=1.0, rep_penalty=1.0,
                 generated_ids=None, rng_key=None):
    logits = np.array(logits, dtype=np.float32)  # transfer from GPU to CPU
```

Sampling runs on the CPU using NumPy (not JAX) because it involves non-differentiable
operations and random number generation that's simpler on CPU.

#### Repetition Penalty

```python
    if rep_penalty != 1.0 and generated_ids:
        seen = list(set(generated_ids))
        for tid in seen:
            if logits[tid] > 0:
                logits[tid] /= rep_penalty    # positive logits: divide (reduce)
            else:
                logits[tid] *= rep_penalty    # negative logits: multiply (push more negative)
```

Previously generated tokens have their logits penalized. A `rep_penalty` of 1.2 means:
- A token with logit 5.0 becomes 5.0 / 1.2 = 4.17 (less likely)
- A token with logit -2.0 becomes -2.0 × 1.2 = -2.4 (even less likely)

This prevents the model from repeating itself excessively.

#### Temperature Scaling

```python
    if temperature != 1.0:
        logits = logits / temperature
```

Temperature controls randomness:
- `temperature < 1.0`: logits become more extreme → more confident, less random
- `temperature > 1.0`: logits become flatter → more random, more creative
- `temperature = 0.0`: special case → greedy (argmax)

Example:
```
logits = [3.0, 1.0, 0.5]

temp=1.0: softmax → [0.67, 0.18, 0.15]  (original distribution)
temp=0.5: [6.0, 2.0, 1.0] → softmax → [0.93, 0.05, 0.02]  (more peaked)
temp=2.0: [1.5, 0.5, 0.25] → softmax → [0.47, 0.27, 0.26]  (more uniform)
```

#### Softmax

```python
    logits -= logits.max()      # subtract max for numerical stability
    probs = np.exp(logits)
    probs /= probs.sum()
```

Standard softmax with the log-sum-exp trick: subtracting the max prevents overflow
when exponentiating large values.

#### Top-p (Nucleus) Sampling

```python
    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)          # sort by probability, descending
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)          # cumulative sum
        cutoff = np.searchsorted(cumsum, top_p) + 1  # find where cumsum reaches top_p
        keep_idx = sorted_idx[:cutoff]            # keep only top tokens
        filtered_probs = probs[keep_idx]
        filtered_probs /= filtered_probs.sum()    # renormalize
        return int(np.random.choice(keep_idx, p=filtered_probs))
```

Top-p keeps only the smallest set of tokens whose combined probability exceeds `top_p`.

Example with top_p=0.9:
```
probs = [0.50, 0.25, 0.15, 0.05, 0.03, 0.02]  (sorted)
cumsum = [0.50, 0.75, 0.90, 0.95, 0.98, 1.00]
cutoff at cumsum >= 0.9 → index 2, keep first 3 tokens
Remaining: [0.50, 0.25, 0.15] → renormalized: [0.556, 0.278, 0.167]
```

The rare tokens (0.05, 0.03, 0.02) are excluded — they can never be sampled.
This prevents the model from generating nonsensical rare tokens while still
allowing some diversity among the likely candidates.

### CLI Entry Point — `main()` at line 150

```python
def main():
    ...
    parser = argparse.ArgumentParser(description="Generate text with Triton kernels")
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--weights", type=str, default="weights.pkl")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--rep-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
```

Model weights are loaded from a pickle file:
```python
    with open(os.path.join(os.path.dirname(__file__), args.weights), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
```

The tokenizer is loaded via `load_bpe_vocab()` from `data.py`:
```python
    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]    # token IDs → text
    tok = Tokenizer.from_file(bpe_vocab["tokenizer_path"])
    encode_fn = lambda text: tok.encode(text).ids     # text → token IDs
```

The prompt is encoded, truncated if necessary, and generation begins:
```python
    prompt_ids = encode_fn(args.prompt)
    if len(prompt_ids) >= ctx_len:
        prompt_ids = prompt_ids[:ctx_len - 1]    # leave room for at least 1 generated token
    max_gen = min(args.max_tokens, ctx_len - len(prompt_ids))
```

#### Streaming Mode (default)

```python
    t0 = time.perf_counter()
    all_tokens = []
    t_first = None
    for token_id in stream_tokens(params, config, prompt_ids, max_gen, ...):
        if t_first is None:
            t_first = time.perf_counter()   # time to first token
        all_tokens.append(token_id)
        new_text = decode_fn([token_id])
        sys.stdout.write(new_text)
        sys.stdout.flush()                  # display each token immediately
```

Each token is decoded and printed as soon as it's generated. The user sees text
appear character by character.

Performance metrics:
```python
    ttft = (t_first - t0) * 1000      # Time To First Token (includes prefill + JIT)
    decode_time = elapsed - (t_first - t0)
    decode_tok_s = (max_gen - 1) / decode_time   # decode throughput
```

**TTFT (Time To First Token)** includes prefill, weight packing, and Triton JIT
compilation (on first run). Subsequent runs reuse compiled kernels.

#### Non-streaming Mode (--no-stream)

```python
    tokens = generate_tokens(params, config, prompt_ids, max_gen, ...)
    text = decode_fn(tokens)
    sys.stdout.write(text)
```

Generates all tokens first, then prints the complete text at once. Slightly faster
because there's no per-token print overhead, but the user sees nothing until generation
completes.

---

## 12. Profiling and Roofline Analysis

`profile_kernels.py` measures decode kernel performance and compares it against
theoretical hardware limits.

### Loading and Setup — Lines 22-123

```python
def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    return params, saved["config"]
```

A fixed 128-token prompt is used for repeatable benchmarks:
```python
    PROMPT_LEN = min(128, ctx)
    _prompt_text = ("The cat sat on the mat and looked at the sky. ...")
    _prompt_ids = _tok.encode(_prompt_text).ids[:PROMPT_LEN]
```

### Prefill Measurement — `measure_prefill()` at line 29

```python
def measure_prefill(params, config, prompt, n_runs=20):
    # 3 warmup runs (trigger JIT, fill caches)
    for _ in range(3):
        logits, kc, vc = prefill_with_kv(params, config, x)
        _ = logits.block_until_ready()

    # n_runs timed measurements
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        logits, kc, vc = prefill_with_kv(params, config, x)
        _ = logits.block_until_ready()
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000    # median ms
```

Warmup runs are critical — the first JAX call triggers XLA compilation. We use
**median** (not mean) to be robust against GC pauses and system jitter.

### Decode Measurement — `measure_decode()` at line 44

```python
def measure_decode(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens=128, n_runs=10):
    # 5 warmup steps (trigger Triton JIT)
    kv_tmp = kv_packed
    t = tok
    for i in range(5):
        t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv_tmp, vocab_size, kv_splits=1)
        _ = int(t)

    # n_runs full generations
    times = []
    for _ in range(n_runs):
        kv_tmp = kv_packed   # reset KV cache each run
        t = tok
        t0 = time.perf_counter()
        for i in range(n_tokens):
            t, _, kv_tmp = multi_sm_decode_nlayer(
                w, config, t, start_pos + i, kv_tmp, vocab_size, kv_splits=1)
            tokens.append(int(t))    # int() forces GPU→CPU sync
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000
```

`int(t)` forces GPU-to-CPU synchronization — without it, JAX would queue the kernel
calls and `time.perf_counter()` would measure only the queuing time, not actual
execution. This is one of the most common benchmarking mistakes with JAX.

### Memory Statistics — `compute_memory_stats()` at line 67

```python
def compute_memory_stats(config):
    per_layer = (d +                    # ln1_scale
                 d*d + d*d_kv +         # Wq + Wk
                 d*d_kv + d*d +         # Wv + Wo
                 d +                     # ln2_scale
                 d*d_ff + d*d_ff +      # gate + up
                 d_ff*d)                # down
    total_weights = vocab_size * d + n_layers * per_layer + d  # +embeddings +final_ln
    weight_bytes = total_weights * 2    # bf16 = 2 bytes per element
    kv_total = n_layers * 2 * n_kv_heads * ctx * d_head * 2
    return {
        "weight_buffer_mb": weight_bytes / 1e6,
        "kv_cache_mb": kv_total / 1e6,
        "total_inference_mb": (weight_bytes + kv_total) / 1e6,
    }
```

### Roofline Analysis — Lines 163-169

```python
    bytes_per_step = (mem["weight_buffer_mb"] + mem["kv_cache_mb"]) * 1e6
    theoretical_min_ms = bytes_per_step / (836e9) * 1000    # 836 GB/s peak bandwidth
    bandwidth_util = theoretical_min_ms / ms_per_tok * 100
```

The **roofline model** gives the theoretical minimum time per token — the time to
simply read all the data at peak bandwidth, assuming perfect memory access patterns
and zero compute overhead.

```
bytes_per_step ≈ 613 MB
theoretical_min = 613 MB / 836 GB/s = 0.73 ms

If we achieve 4.3 ms/tok → bandwidth_util = 0.73 / 4.3 × 100 = 17%
```

17% bandwidth utilization means we're spending 83% of time on overhead: barriers,
work imbalance, memory access inefficiencies, and L2 cache misses. The roofline
tells us how much room there is for improvement — in theory, we could be ~6x faster
if we achieved perfect bandwidth utilization.

### Profiling Output

The profiler prints a complete performance summary:

```
============================================================
KERNEL PROFILING
============================================================
Model:    d=1024 h=16 l=24 ctx=512
Params:   303,350,784
Generate: 128 tokens from 128-token prompt
GPU:      NVIDIA GeForce RTX 4080 SUPER
Weights:  607.2 MB, KV cache: 6.3 MB

--- Prefill (128 tokens, JAX) ---
  157.3 ms (814 tok/s)

--- Decode (128 tokens, Triton multi-SM, grid=16) ---
  553.4 ms total, 4.323 ms/tok, 231 tok/s

--- End-to-End ---
  710.7 ms for 256 tokens
  Text: [generated text preview]...

--- Roofline ---
  619.3 MB per step, theoretical min 0.741 ms/tok
  Achieved 4.323 ms/tok = 17% bandwidth utilization
```
