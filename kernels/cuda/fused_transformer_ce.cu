/**
 * Fused transformer forward pass + CE loss CUDA kernel for EGGROLL ES training.
 *
 * This kernel replaces the Triton version (kernels/fused_transformer_ce.py).
 * Key advantage: explicit shared memory control enables FlashAttention-style
 * attention tiling, which the Triton kernel cannot do because Triton lacks
 * the ability to slice register-resident 2D tensors by rows.
 *
 * Architecture: d_model=64, n_heads=2, d_head=32, d_ff=256, vocab=65, seq=128.
 *
 * Grid:  (HALF_POP, BATCH / BATCH_TILE, 2)  — one block per (perturbation, batch_tile, sign)
 * Block: 128 or 256 threads (4 or 8 warps)
 *
 * Shared memory layout (per block):
 *   K[SEQ][D_HEAD]  — 128*32*4 = 16KB per head
 *   V[SEQ][D_HEAD]  — 128*32*4 = 16KB per head
 *   Total: 32KB per head, 64KB for 2 heads (fits in 100KB shared mem on Ada)
 *
 * The FlashAttention tiling strategy:
 *   1. Compute full K, V for the sequence and store to shared memory
 *   2. For each query tile (BLOCK_Q rows of Q):
 *      a. Compute Q_tile from h_norm (recompute per tile)
 *      b. For each KV tile:
 *         - Load K_tile, V_tile from shared memory
 *         - scores_tile = Q_tile @ K_tile^T  (BLOCK_Q x BLOCK_KV)
 *         - Apply causal mask
 *         - Online softmax update
 *         - Accumulate attention output
 *      c. Finalize attention output for this query tile
 *   3. Continue with O-projection, FFN, output proj, CE loss
 *
 * Build: make -C kernels/cuda/
 * Test:  uv run kernels/cuda/test_kernel.py
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>  // for wmma tensor core operations
#include <cstdint>
#include <cmath>
#include <cfloat>

// ─── Architecture constants ───
constexpr int SEQ = 128;
constexpr int D_MODEL = 64;
constexpr int D_HEAD = 32;
constexpr int N_HEADS = 2;
constexpr int D_FF = 256;
constexpr int VOCAB = 65;
constexpr int VOCAB_PAD = 128;  // padded to power of 2

// ─── Tiling constants (tune these) ───
constexpr int BLOCK_Q = 32;    // query tile size for FlashAttention
constexpr int BLOCK_KV = 32;   // key/value tile size for FlashAttention
constexpr int BLOCK_FF = 32;   // FFN tiling (matches Triton BLOCK_K)
constexpr int BATCH_TILE = 1;  // batch elements per block (start with 1, try 2/4 later)

// ─── Perturbation vector offsets (must match build_param_spec alphabetical order) ───
constexpr int OFF_AK   = 0;     // layer0.attn.k (64,64): b(64)+a(64)
constexpr int OFF_AO   = 128;   // layer0.attn.o (64,64): b(64)+a(64)
constexpr int OFF_AQ   = 256;   // layer0.attn.q (64,64): b(64)+a(64)
constexpr int OFF_AV   = 384;   // layer0.attn.v (64,64): b(64)+a(64)
constexpr int OFF_FD   = 512;   // layer0.ffn.down (256,64): b(256)+a(64)
constexpr int OFF_FDB  = 832;   // layer0.ffn.down_bias (64,)
constexpr int OFF_FU   = 896;   // layer0.ffn.up (64,256): b(64)+a(256)
constexpr int OFF_FUB  = 1216;  // layer0.ffn.up_bias (256,)
constexpr int OFF_LN1B = 1472;  // layer0.ln1.bias (64,)
constexpr int OFF_LN1S = 1536;  // layer0.ln1.scale (64,)
constexpr int OFF_LN2B = 1600;  // layer0.ln2.bias (64,)
constexpr int OFF_LN2S = 1664;  // layer0.ln2.scale (64,)
constexpr int OFF_LNFB = 1728;  // ln_final.bias (64,)
constexpr int OFF_LNFS = 1792;  // ln_final.scale (64,)
constexpr int OFF_OP   = 1856;  // output_proj (64,65): b(64)+a(65)
constexpr int OFF_PE   = 1985;  // pos_emb (128,64): b(128)+a(64)
constexpr int OFF_TE   = 2177;  // token_emb (65,64): b(65)+a(64)
constexpr int VEC_DIM  = 2306;

// ─── Shared memory struct ───
// Allocate K and V for one head at a time (reuse across heads)
struct SharedMem {
    float K[SEQ][D_HEAD];    // 128*32 = 4096 floats = 16KB
    float V[SEQ][D_HEAD];    // 128*32 = 4096 floats = 16KB
    // Total: 32KB — well within 100KB limit
    // Can add more buffers here if needed (e.g., for h_norm tiles)
};

// ─── Helper: bf16 matrix load ───
__device__ inline float load_bf16_as_f32(const __nv_bfloat16* ptr, int idx) {
    return __bfloat162float(ptr[idx]);
}

// ─── Helper: GELU approximation (matches Triton's sigmoid-based GELU) ───
__device__ inline float gelu_approx(float x) {
    // GELU(x) = x * sigmoid(1.702 * x)
    float s = 1.0f / (1.0f + expf(-1.702f * x));
    return x * s;
}

// ─── Helper: online softmax for FlashAttention ───
// Updates running (max, sum, output) for a new tile of attention scores
// m_old: previous row-wise max, l_old: previous row-wise sum of exp
// scores: (BLOCK_Q, BLOCK_KV) tile of attention scores
// V_tile: (BLOCK_KV, D_HEAD) tile of values
// o: (BLOCK_Q, D_HEAD) running output accumulator
//
// After update:
//   m_new = max(m_old, max(scores))
//   l_new = l_old * exp(m_old - m_new) + sum(exp(scores - m_new))
//   o_new = o * exp(m_old - m_new) + exp(scores - m_new) @ V_tile
//
// This is the core FlashAttention algorithm (Dao et al., 2022).
// TODO: Implement this function — it's the key to reducing register pressure


/**
 * Main fused transformer + CE loss kernel.
 *
 * TODO: Implement the full kernel. The structure should be:
 *
 * 1. Load perturbation vector components (shared across batch tile)
 * 2. For each batch element in BATCH_TILE:
 *    a. Load tokens (x_seq) and compute embedding + perturbation → h
 *    b. LayerNorm 1 → h_norm
 *    c. For each attention head:
 *       - Compute K, V from h_norm and store to shared memory
 *       - __syncthreads()
 *       - FlashAttention: tile over query positions, for each Q tile:
 *         * Compute Q_tile from h_norm rows
 *         * Tile over KV positions, load K_tile/V_tile from shared mem
 *         * Online softmax + accumulate attention output
 *       - O projection + residual connection
 *    d. LayerNorm 2
 *    e. FFN (tiled, same as Triton version)
 *    f. Final LayerNorm
 *    g. Output projection + label-smoothed CE loss
 *    h. Store CE to output buffer
 *
 * Grid dimensions:
 *   blockIdx.x = perturbation member index (0..HALF_POP-1)
 *   blockIdx.y = batch tile index (0..BATCH/BATCH_TILE-1)
 *   blockIdx.z = sign (0=+sigma, 1=-sigma)
 *
 * Thread mapping (for 128 threads / 4 warps):
 *   Each thread handles multiple elements of the sequence/feature dimensions.
 *   For (SEQ=128, D_MODEL=64): 128*64 = 8192 elements / 128 threads = 64 elements/thread
 *   Use thread ID to assign rows: thread i handles rows [i*stride, i*stride+stride)
 */
__global__ void fused_transformer_ce_kernel(
    // Base weights (bf16, row-major)
    const __nv_bfloat16* __restrict__ token_emb,   // [VOCAB, D_MODEL]
    const __nv_bfloat16* __restrict__ pos_emb,     // [SEQ, D_MODEL]
    const __nv_bfloat16* __restrict__ ln1_scale,   // [D_MODEL]
    const __nv_bfloat16* __restrict__ ln1_bias,    // [D_MODEL]
    const __nv_bfloat16* __restrict__ wq,          // [D_MODEL, D_MODEL]
    const __nv_bfloat16* __restrict__ wk,          // [D_MODEL, D_MODEL]
    const __nv_bfloat16* __restrict__ wv,          // [D_MODEL, D_MODEL]
    const __nv_bfloat16* __restrict__ wo,          // [D_MODEL, D_MODEL]
    const __nv_bfloat16* __restrict__ ln2_scale,   // [D_MODEL]
    const __nv_bfloat16* __restrict__ ln2_bias,    // [D_MODEL]
    const __nv_bfloat16* __restrict__ ffn_up,      // [D_MODEL, D_FF]
    const __nv_bfloat16* __restrict__ ffn_up_bias, // [D_FF]
    const __nv_bfloat16* __restrict__ ffn_down,    // [D_FF, D_MODEL]
    const __nv_bfloat16* __restrict__ ffn_down_bias, // [D_MODEL]
    const __nv_bfloat16* __restrict__ ln_final_scale, // [D_MODEL]
    const __nv_bfloat16* __restrict__ ln_final_bias,  // [D_MODEL]
    const __nv_bfloat16* __restrict__ output_proj, // [D_MODEL, VOCAB_PAD]
    // Perturbation vectors (f32)
    const float* __restrict__ vecs,                // [HALF_POP, VEC_DIM]
    // Input data
    const int32_t* __restrict__ x,                 // [BATCH, SEQ]
    const int32_t* __restrict__ y,                 // [BATCH, SEQ]
    // Scalars
    float sigma,
    float alpha,
    float temperature,
    // Output
    float* __restrict__ ce_pos,                    // [HALF_POP, BATCH]
    float* __restrict__ ce_neg,                    // [HALF_POP, BATCH]
    // Grid info
    int half_pop,
    int batch
) {
    const int pid_p = blockIdx.x;
    const int pid_bt = blockIdx.y;
    const int pid_sign = blockIdx.z;
    const float sign_sigma = (pid_sign == 0) ? sigma : -sigma;

    // Perturbation vector base offset for this member
    const float* vec = vecs + pid_p * VEC_DIM;

    // Shared memory for K, V (per-head, reused)
    extern __shared__ char smem_raw[];
    SharedMem* smem = reinterpret_cast<SharedMem*>(smem_raw);

    // TODO: Implement the full forward pass here.
    // See the Triton kernel (kernels/fused_transformer_ce.py) for the computation.
    // The key difference: use shared memory for K, V to enable FlashAttention tiling.
    //
    // Start by implementing without FlashAttention (full 128x128 attention in registers)
    // to verify correctness against the Triton kernel. Then add tiling.

    // Placeholder: write zero CE
    const int actual_b = pid_bt * BATCH_TILE;
    float* out_ptr = (pid_sign == 0) ? ce_pos : ce_neg;
    if (threadIdx.x == 0) {
        out_ptr[pid_p * batch + actual_b] = 0.0f;
    }
}


// ─── XLA custom call interface ───
// JAX calls this function via xla_client.register_custom_call_target.
// The buffer layout follows XLA's custom call convention:
//   buffers[0..16]: input weight arrays (bf16)
//   buffers[17]: perturbation vectors (f32)
//   buffers[18]: x tokens (int32)
//   buffers[19]: y labels (int32)
//   buffers[20]: sigma (f32 scalar)
//   buffers[21]: alpha (f32 scalar)
//   buffers[22]: temperature (f32 scalar)
//   buffers[23]: ce_pos output (f32)
//   buffers[24]: ce_neg output (f32)
//
// opaque: packed struct with {half_pop, batch} as int32s
struct KernelParams {
    int32_t half_pop;
    int32_t batch;
};

extern "C" void fused_transformer_ce_cuda(
    cudaStream_t stream,
    void** buffers,
    const char* opaque,
    size_t opaque_len
) {
    // Unpack parameters
    const KernelParams* params = reinterpret_cast<const KernelParams*>(opaque);
    const int half_pop = params->half_pop;
    const int batch = params->batch;

    // Unpack input buffers
    const __nv_bfloat16* token_emb     = reinterpret_cast<const __nv_bfloat16*>(buffers[0]);
    const __nv_bfloat16* pos_emb       = reinterpret_cast<const __nv_bfloat16*>(buffers[1]);
    const __nv_bfloat16* ln1_scale     = reinterpret_cast<const __nv_bfloat16*>(buffers[2]);
    const __nv_bfloat16* ln1_bias      = reinterpret_cast<const __nv_bfloat16*>(buffers[3]);
    const __nv_bfloat16* wq            = reinterpret_cast<const __nv_bfloat16*>(buffers[4]);
    const __nv_bfloat16* wk            = reinterpret_cast<const __nv_bfloat16*>(buffers[5]);
    const __nv_bfloat16* wv            = reinterpret_cast<const __nv_bfloat16*>(buffers[6]);
    const __nv_bfloat16* wo            = reinterpret_cast<const __nv_bfloat16*>(buffers[7]);
    const __nv_bfloat16* ln2_scale     = reinterpret_cast<const __nv_bfloat16*>(buffers[8]);
    const __nv_bfloat16* ln2_bias      = reinterpret_cast<const __nv_bfloat16*>(buffers[9]);
    const __nv_bfloat16* ffn_up        = reinterpret_cast<const __nv_bfloat16*>(buffers[10]);
    const __nv_bfloat16* ffn_up_bias   = reinterpret_cast<const __nv_bfloat16*>(buffers[11]);
    const __nv_bfloat16* ffn_down      = reinterpret_cast<const __nv_bfloat16*>(buffers[12]);
    const __nv_bfloat16* ffn_down_bias = reinterpret_cast<const __nv_bfloat16*>(buffers[13]);
    const __nv_bfloat16* ln_final_scale = reinterpret_cast<const __nv_bfloat16*>(buffers[14]);
    const __nv_bfloat16* ln_final_bias = reinterpret_cast<const __nv_bfloat16*>(buffers[15]);
    const __nv_bfloat16* output_proj   = reinterpret_cast<const __nv_bfloat16*>(buffers[16]);
    const float* vecs                  = reinterpret_cast<const float*>(buffers[17]);
    const int32_t* x                   = reinterpret_cast<const int32_t*>(buffers[18]);
    const int32_t* y                   = reinterpret_cast<const int32_t*>(buffers[19]);
    const float sigma       = *reinterpret_cast<const float*>(buffers[20]);
    const float alpha_val   = *reinterpret_cast<const float*>(buffers[21]);
    const float temperature = *reinterpret_cast<const float*>(buffers[22]);

    // Output buffers
    float* ce_pos = reinterpret_cast<float*>(buffers[23]);
    float* ce_neg = reinterpret_cast<float*>(buffers[24]);

    // Launch kernel
    dim3 grid(half_pop, batch / BATCH_TILE, 2);
    dim3 block(128);  // 4 warps
    size_t smem_size = sizeof(SharedMem);

    fused_transformer_ce_kernel<<<grid, block, smem_size, stream>>>(
        token_emb, pos_emb, ln1_scale, ln1_bias,
        wq, wk, wv, wo,
        ln2_scale, ln2_bias,
        ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
        ln_final_scale, ln_final_bias, output_proj,
        vecs, x, y,
        sigma, alpha_val, temperature,
        ce_pos, ce_neg,
        half_pop, batch
    );
}
