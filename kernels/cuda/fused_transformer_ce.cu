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
 * Block: 128 threads (4 warps)
 *
 * WMMA tensor core acceleration:
 *   Uses nvcuda::wmma 16x16x16 bf16->f32 tiles for all large matmuls:
 *   - QKV projections: (128,64) x (64,32) -> (128,32)
 *   - O projection: (128,32) x (32,64) -> (128,64)
 *   - FFN up: (128,64) x (64,32) -> (128,32) per tile
 *   - FFN down: (128,32) x (32,64) -> (128,64) per tile
 *   - Output proj: (128,64) x (64,128) -> (128,128)
 *   FlashAttention and perturbation corrections remain scalar.
 *
 * Shared memory layout (per block, using unions for phase reuse):
 *   WMMA phase: A_bf16[128][64]=16KB + B_bf16[64][32]=4KB + K[128][32]=16KB + V[128][32]=16KB
 *   Peak: ~52.5KB (during V computation when A, B, K, V all live)
 *   Requires cudaFuncSetAttribute for >48KB shared memory.
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

using namespace nvcuda;

// ─── Architecture constants ───
constexpr int SEQ = 128;
constexpr int D_MODEL = 64;
constexpr int D_HEAD = 32;
constexpr int N_HEADS = 2;
constexpr int D_FF = 256;
constexpr int VOCAB = 65;
constexpr int VOCAB_PAD = 128;  // padded to power of 2

// ─── WMMA tile dimensions ───
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr int NUM_WARPS = 4;

// ─── Tiling constants (tune these) ───
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
// Layout designed for WMMA matmul phases + FlashAttention reuse.
// Peak usage ~52.5KB occurs during V computation (A + B + K + V all live).
//
// Memory map (byte offsets):
//   0:     A_bf16[128][64]  = 16384 bytes (WMMA input matrix, row-major)
//   16384: B_bf16[64][32]   = 4096 bytes  (WMMA weight tile, row-major)
//          -- B can also hold [64][64]=8192 or [32][64]=4096 depending on matmul
//   20480: K/C1[128][32]    = 16384 bytes (attention K or WMMA output)
//   36864: V/C2[128][32]    = 16384 bytes (attention V or WMMA output)
//   53248: reduce[128]      = 512 bytes   (CE loss reduction)
//   Total: 53760 bytes = 52.5KB
//
// For wider outputs (128x64 or 128x128), C1+C2 are used together as one buffer.

struct SharedMem {
    __nv_bfloat16 A[SEQ][D_MODEL];        // 128*64 = 8192 bf16 = 16KB
    __nv_bfloat16 B[D_MODEL][D_HEAD];     // 64*32 = 2048 bf16 = 4KB (smallest config)
    float C1[SEQ][D_HEAD];                 // 128*32 = 4096 floats = 16KB
    float C2[SEQ][D_HEAD];                 // 128*32 = 4096 floats = 16KB
    float reduce[SEQ];                     // 128 floats = 512B
};

// Larger B overlays (reinterpreted from SharedMem.B start):
// B_wide[D_MODEL][D_MODEL]   = 64*64*2 = 8KB (for O proj, FFN down, output proj weight tiles)
// B_tall[D_HEAD][D_MODEL]    = 32*64*2 = 4KB (same as B but transposed interpretation)

// Accessor: get pointer to B region reinterpreted as [rows][cols] bf16
// (B starts at offset sizeof(A) = 16384 bytes into SharedMem)

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

// ─── WMMA matmul: (128, K_dim) x (K_dim, N_dim) -> output in shared memory ───
// A is in smem->A (bf16, row-major, 128 x K_dim, padded to 128x64)
// B is in smem->B area (bf16, row-major, K_dim x N_dim)
// C is output (f32), stored to smem at out_ptr (row-major, 128 x N_dim)
//
// This function is called by ALL threads in the block. Each warp handles
// a subset of the output tiles.
//
// Template params:
//   K_DIM: inner dimension (16, 32, or 64)
//   N_DIM: output columns (32 or 64)
//   A_STRIDE: row stride of A in shared memory (elements, typically D_MODEL=64)
//   B_STRIDE: row stride of B in shared memory (elements, typically N_DIM)
template<int K_DIM, int N_DIM, int A_STRIDE, int B_STRIDE>
__device__ void wmma_matmul_128xKxN(
    const __nv_bfloat16* A_smem,  // [128][A_STRIDE], only first K_DIM cols used
    const __nv_bfloat16* B_smem,  // [K_DIM][B_STRIDE], only first N_DIM cols used
    float* C_smem,                 // [128][N_DIM] output
    int warp_id
) {
    // Output tile grid: M_tiles x N_tiles, where M_tiles = 128/16 = 8, N_tiles = N_DIM/16
    constexpr int M_TILES = SEQ / WMMA_M;         // 8
    constexpr int N_TILES = N_DIM / WMMA_N;       // 2 for N=32, 4 for N=64
    constexpr int K_TILES = K_DIM / WMMA_K;       // 1,2, or 4
    constexpr int TOTAL_OUT_TILES = M_TILES * N_TILES;  // 16 or 32

    // Distribute output tiles across warps
    // Each warp handles TOTAL_OUT_TILES / NUM_WARPS tiles
    constexpr int TILES_PER_WARP = TOTAL_OUT_TILES / NUM_WARPS;

    for (int t = 0; t < TILES_PER_WARP; t++) {
        int tile_idx = warp_id * TILES_PER_WARP + t;
        int m_tile = tile_idx / N_TILES;  // which 16-row block
        int n_tile = tile_idx % N_TILES;  // which 16-col block

        // Declare accumulator fragment
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
        wmma::fill_fragment(acc, 0.0f);

        // Accumulate over K tiles
        for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
            // Load A fragment: rows [m_tile*16, m_tile*16+16), cols [k_tile*16, k_tile*16+16)
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                          __nv_bfloat16, wmma::row_major> a_frag;
            const __nv_bfloat16* a_ptr = A_smem + m_tile * WMMA_M * A_STRIDE + k_tile * WMMA_K;
            wmma::load_matrix_sync(a_frag, a_ptr, A_STRIDE);

            // Load B fragment: rows [k_tile*16, k_tile*16+16), cols [n_tile*16, n_tile*16+16)
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                          __nv_bfloat16, wmma::row_major> b_frag;
            const __nv_bfloat16* b_ptr = B_smem + k_tile * WMMA_K * B_STRIDE + n_tile * WMMA_N;
            wmma::load_matrix_sync(b_frag, b_ptr, B_STRIDE);

            // Accumulate
            wmma::mma_sync(acc, a_frag, b_frag, acc);
        }

        // Store accumulator to shared memory output
        float* c_ptr = C_smem + m_tile * WMMA_M * N_DIM + n_tile * WMMA_N;
        wmma::store_matrix_sync(c_ptr, acc, N_DIM, wmma::mem_row_major);
    }
}


/**
 * Main fused transformer + CE loss kernel.
 *
 * Thread mapping: 128 threads (4 warps). Thread tid owns sequence position tid.
 * Each thread computes one row of all (128, 64) or (128, 32) matrices.
 *
 * WMMA matmuls use shared memory for input/output; per-thread register values
 * are written to shared memory before WMMA and read back after.
 *
 * Register budget per thread:
 *   h[64], h_norm[64], Q[32], attn_out[32] + scalars ~= 200 regs
 *   K and V live in shared memory (FlashAttention style).
 *
 * Grid dimensions:
 *   blockIdx.x = perturbation member index (0..HALF_POP-1)
 *   blockIdx.y = batch tile index (0..BATCH/BATCH_TILE-1)
 *   blockIdx.z = sign (0=+sigma, 1=-sigma)
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

    // Thread ID = sequence position this thread owns
    const int tid = threadIdx.x;  // 0..127
    const int warp_id = tid / 32;
    // const int lane_id = tid % 32;  // unused for now

    // Perturbation vector base pointer for this member
    const float* vec = vecs + pid_p * VEC_DIM;

    // Shared memory
    extern __shared__ char smem_raw[];
    SharedMem* smem = reinterpret_cast<SharedMem*>(smem_raw);

    // Batch element index
    const int actual_b = pid_bt * BATCH_TILE;

    // ════════════════════════════════════════════════════════════════════
    // Phase 1: Embedding + perturbation -> h[D_MODEL]
    // ════════════════════════════════════════════════════════════════════
    // Each thread loads its own token and position embedding row
    const int token_id = x[actual_b * SEQ + tid];

    float h[D_MODEL];
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        float emb = load_bf16_as_f32(token_emb, token_id * D_MODEL + d);
        float pos = load_bf16_as_f32(pos_emb, tid * D_MODEL + d);
        h[d] = emb + pos;
    }

    // Token embedding perturbation: W_pert = W + sign_sigma * outer(b_te, a_te)
    // For row token_id: emb_pert[d] = sign_sigma * b_te[token_id] * a_te[d]
    float b_te = vec[OFF_TE + token_id];
    float b_pe = vec[OFF_PE + tid];
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        float a_te = vec[OFF_TE + VOCAB + d];
        float a_pe = vec[OFF_PE + SEQ + d];
        h[d] += sign_sigma * (b_te * a_te + b_pe * a_pe);
    }

    // ════════════════════════════════════════════════════════════════════
    // Phase 2: LayerNorm 1 -> h_norm[D_MODEL]
    // ════════════════════════════════════════════════════════════════════
    float mean = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) mean += h[d];
    mean /= D_MODEL;

    float h_norm[D_MODEL];
    float var = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h_norm[d] = h[d] - mean;
        var += h_norm[d] * h_norm[d];
    }
    var /= D_MODEL;
    float inv_std = rsqrtf(var + 1e-5f);

    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        float s = load_bf16_as_f32(ln1_scale, d) + sign_sigma * vec[OFF_LN1S + d];
        float b = load_bf16_as_f32(ln1_bias, d) + sign_sigma * vec[OFF_LN1B + d];
        h_norm[d] = s * h_norm[d] * inv_std + b;
    }

    // ════════════════════════════════════════════════════════════════════
    // Phase 3: Multi-head attention (WMMA matmuls + FlashAttention)
    // ════════════════════════════════════════════════════════════════════
    // Precompute dot products for perturbation: h_norm . b_q, h_norm . b_k, h_norm . b_v
    float h_dot_bq = 0.0f, h_dot_bk = 0.0f, h_dot_bv = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h_dot_bq += h_norm[d] * vec[OFF_AQ + d];
        h_dot_bk += h_norm[d] * vec[OFF_AK + d];
        h_dot_bv += h_norm[d] * vec[OFF_AV + d];
    }

    // Load a_o perturbation vector (shared across heads, applied after both heads)
    float a_o[D_MODEL];
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        a_o[d] = vec[OFF_AO + D_MODEL + d];
    }
    float o_pert = 0.0f;  // accumulate across heads: sum(attn_out * b_o_h)

    constexpr float scale = 0.17677669529663689f;  // 1/sqrt(32) = 1/sqrt(D_HEAD)

    // ── Write h_norm to shared memory A as bf16 for WMMA ──
    // This is done once before the head loop; h_norm is the same for both heads.
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        smem->A[tid][d] = __float2bfloat16(h_norm[d]);
    }
    __syncthreads();

    for (int head = 0; head < N_HEADS; head++) {
        const int head_off = head * D_HEAD;

        // ── Load wk_head columns to shared memory B as bf16 ──
        // wk is [D_MODEL, D_MODEL], we need columns [head_off..head_off+D_HEAD)
        // = rows 0..63 of wk, columns head_off..head_off+31
        // B layout: [D_MODEL][D_HEAD] = [64][32] row-major
        // Cooperative load: 128 threads load 64*32=2048 elements
        {
            constexpr int TOTAL_ELEMS = D_MODEL * D_HEAD;  // 2048
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / SEQ;  // 16
            int base = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                int idx = base + i;
                int row = idx / D_HEAD;
                int col = idx % D_HEAD;
                smem->B[row][col] = wk[row * D_MODEL + head_off + col];
            }
        }
        __syncthreads();

        // ── WMMA: K = A @ B = h_norm(128,64) @ wk_head(64,32) -> C1(128,32) ──
        wmma_matmul_128xKxN<D_MODEL, D_HEAD, D_MODEL, D_HEAD>(
            &smem->A[0][0], &smem->B[0][0], &smem->C1[0][0], warp_id);
        __syncthreads();

        // ── Read K from C1 and add perturbation, store to C1 (which serves as K) ──
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            float a_k_dh = vec[OFF_AK + D_MODEL + head_off + dh];
            smem->C1[tid][dh] += sign_sigma * h_dot_bk * a_k_dh;
        }
        __syncthreads();

        // ── Load wv_head columns to shared memory B ──
        {
            constexpr int TOTAL_ELEMS = D_MODEL * D_HEAD;
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / SEQ;
            int base = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                int idx = base + i;
                int row = idx / D_HEAD;
                int col = idx % D_HEAD;
                smem->B[row][col] = wv[row * D_MODEL + head_off + col];
            }
        }
        __syncthreads();

        // ── WMMA: V = A @ B = h_norm(128,64) @ wv_head(64,32) -> C2(128,32) ──
        wmma_matmul_128xKxN<D_MODEL, D_HEAD, D_MODEL, D_HEAD>(
            &smem->A[0][0], &smem->B[0][0], &smem->C2[0][0], warp_id);
        __syncthreads();

        // ── Read V from C2 and add perturbation, store back to C2 (which serves as V) ──
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            float a_v_dh = vec[OFF_AV + D_MODEL + head_off + dh];
            smem->C2[tid][dh] += sign_sigma * h_dot_bv * a_v_dh;
        }
        __syncthreads();

        // At this point:
        // smem->C1 = K[128][32] (float)
        // smem->C2 = V[128][32] (float)
        // smem->A still holds h_norm (bf16) -- we can reuse for Q computation

        // ── Compute Q via WMMA, store temporarily, read back to registers ──
        // Load wq_head to B
        {
            constexpr int TOTAL_ELEMS = D_MODEL * D_HEAD;
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / SEQ;
            int base = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                int idx = base + i;
                int row = idx / D_HEAD;
                int col = idx % D_HEAD;
                smem->B[row][col] = wq[row * D_MODEL + head_off + col];
            }
        }
        __syncthreads();

        // We need a temp buffer for Q output. We can't use C1 or C2 (they hold K, V).
        // But we only need thread tid's row (32 values) in registers.
        // Strategy: compute Q matmul into a temp region that overlaps with A
        // (we're done reading A for this head's projections).
        // Actually, we need A intact if we're going to reuse it for the next head's
        // K/V/Q projections. But h_norm doesn't change between heads -- we wrote it
        // once before the loop. So after Q WMMA, A is still valid for next head.
        //
        // Problem: we have no temp space. C1=K, C2=V, A=h_norm (needed for next head).
        //
        // Solution: each thread computes Q via scalar (register-only, no smem needed).
        // Q is small (32 values per thread) and only needed in registers.
        // The cost is 128*64*32 = 262K FMAs scalar. This is acceptable since
        // Q computation is overlapped with K/V memory traffic, and it avoids
        // the shared memory conflict entirely.
        float Q[D_HEAD];
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            float q_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_MODEL; d++) {
                q_val += h_norm[d] * load_bf16_as_f32(wq, d * D_MODEL + head_off + dh);
            }
            float a_q_dh = vec[OFF_AQ + D_MODEL + head_off + dh];
            q_val += sign_sigma * h_dot_bq * a_q_dh;
            Q[dh] = q_val;
        }

        // ── FlashAttention: online softmax over KV tiles ──
        // K is in smem->C1, V is in smem->C2
        float m_running = -FLT_MAX;
        float l_running = 0.0f;
        float o_acc[D_HEAD];
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) o_acc[dh] = 0.0f;

        for (int kv_start = 0; kv_start < SEQ; kv_start += BLOCK_KV) {
            float scores[BLOCK_KV];
            float tile_max = -FLT_MAX;

            #pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                int kv_pos = kv_start + j;
                if (kv_pos > tid) {
                    scores[j] = -FLT_MAX;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        dot += Q[dh] * smem->C1[kv_pos][dh];  // K in C1
                    }
                    scores[j] = dot * scale;
                }
                if (scores[j] > tile_max) tile_max = scores[j];
            }

            float m_new = fmaxf(m_running, tile_max);
            float correction = expf(m_running - m_new);

            l_running *= correction;
            #pragma unroll
            for (int dh = 0; dh < D_HEAD; dh++) {
                o_acc[dh] *= correction;
            }

            #pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                float p = expf(scores[j] - m_new);
                l_running += p;
                int kv_pos = kv_start + j;
                #pragma unroll
                for (int dh = 0; dh < D_HEAD; dh++) {
                    o_acc[dh] += p * smem->C2[kv_pos][dh];  // V in C2
                }
            }

            m_running = m_new;
        }

        // Finalize attention output
        float inv_l = 1.0f / l_running;
        float attn_out[D_HEAD];
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            attn_out[dh] = o_acc[dh] * inv_l;
        }

        // ── O projection: h += attn_out @ wo_head ──
        // attn_out[D_HEAD=32] per thread, wo_head is [D_HEAD, D_MODEL] = [32, 64]
        // Output is h[D_MODEL=64] per thread.
        //
        // For WMMA: (128,32) x (32,64) -> (128,64)
        // Write attn_out to smem A[tid][0..31] as bf16 (reuse A, no longer need h_norm
        // for this head's projections -- but we need it for the NEXT head).
        // Actually, we need h_norm in A for the next head iteration. So we can't
        // overwrite A with attn_out.
        //
        // Alternative: use scalar O projection. It's (128*32*64) = 262K FMAs, same as Q.
        // The O projection reads from global memory (wo) which is likely in L2 cache.
        #pragma unroll
        for (int d = 0; d < D_MODEL; d++) {
            float proj = 0.0f;
            #pragma unroll
            for (int dh = 0; dh < D_HEAD; dh++) {
                proj += attn_out[dh] * load_bf16_as_f32(wo, (head_off + dh) * D_MODEL + d);
            }
            h[d] += proj;
        }

        // ── O projection perturbation accumulation ──
        float b_o_dot = 0.0f;
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            b_o_dot += attn_out[dh] * vec[OFF_AO + head_off + dh];
        }
        o_pert += b_o_dot;

        // Need syncthreads before next head reuses shared memory
        __syncthreads();
    }

    // Apply O projection perturbation: h += sign_sigma * o_pert * a_o
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h[d] += sign_sigma * o_pert * a_o[d];
    }

    // ════════════════════════════════════════════════════════════════════
    // Phase 4: LayerNorm 2
    // ════════════════════════════════════════════════════════════════════
    float mean2 = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) mean2 += h[d];
    mean2 /= D_MODEL;

    float h_norm2[D_MODEL];
    float var2 = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h_norm2[d] = h[d] - mean2;
        var2 += h_norm2[d] * h_norm2[d];
    }
    var2 /= D_MODEL;
    float inv_std2 = rsqrtf(var2 + 1e-5f);

    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        float s = load_bf16_as_f32(ln2_scale, d) + sign_sigma * vec[OFF_LN2S + d];
        float b = load_bf16_as_f32(ln2_bias, d) + sign_sigma * vec[OFF_LN2B + d];
        h_norm2[d] = s * h_norm2[d] * inv_std2 + b;
    }

    // ════════════════════════════════════════════════════════════════════
    // Phase 5: FFN (WMMA for up/down projections, tiled over D_FF)
    // ════════════════════════════════════════════════════════════════════
    // Precompute h_norm2 . b_fu for up-projection perturbation
    float hn2_dot_bfu = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        hn2_dot_bfu += h_norm2[d] * vec[OFF_FU + d];
    }

    // a_fd vector for down-projection perturbation (applied at the end)
    float a_fd[D_MODEL];
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        a_fd[d] = vec[OFF_FD + D_FF + d];
    }

    // Accumulators for FFN output
    float ffn_down_accum[D_MODEL];
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) ffn_down_accum[d] = 0.0f;
    float h_dot_bfd = 0.0f;

    // ── Write h_norm2 to shared memory A as bf16 for WMMA ──
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        smem->A[tid][d] = __float2bfloat16(h_norm2[d]);
    }
    __syncthreads();

    for (int k = 0; k < D_FF; k += BLOCK_FF) {
        // ── FFN Up: h_norm2(128,64) @ ffn_up_tile(64,32) -> up_tile(128,32) via WMMA ──

        // Load ffn_up weight tile to B: ffn_up[:, k..k+32] = [D_MODEL][BLOCK_FF]
        {
            constexpr int TOTAL_ELEMS = D_MODEL * BLOCK_FF;  // 2048
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / SEQ;  // 16
            int base = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                int idx = base + i;
                int row = idx / BLOCK_FF;
                int col = idx % BLOCK_FF;
                smem->B[row][col] = ffn_up[row * D_FF + k + col];
            }
        }
        __syncthreads();

        // WMMA matmul -> C1 (128x32)
        wmma_matmul_128xKxN<D_MODEL, D_HEAD, D_MODEL, D_HEAD>(
            &smem->A[0][0], &smem->B[0][0], &smem->C1[0][0], warp_id);
        __syncthreads();

        // Read up_tile from C1, add perturbation + bias, apply GELU
        float up_tile[BLOCK_FF];
        #pragma unroll
        for (int kk = 0; kk < BLOCK_FF; kk++) {
            int ff_idx = k + kk;
            float val = smem->C1[tid][kk];
            // Perturbation
            float a_fu_k = vec[OFF_FU + D_MODEL + ff_idx];
            val += sign_sigma * hn2_dot_bfu * a_fu_k;
            // Bias + bias perturbation
            val += load_bf16_as_f32(ffn_up_bias, ff_idx);
            val += sign_sigma * vec[OFF_FUB + ff_idx];
            // GELU activation
            up_tile[kk] = gelu_approx(val);
        }

        // ── FFN Down: up_tile(128,32) @ ffn_down_tile(32,64) -> accum(128,64) via WMMA ──
        // Write up_tile to smem A[tid][0..31] as bf16 for WMMA input
        // But A is [128][64] and we only need [128][32]. Use the first 32 columns.
        // Actually, WMMA K_DIM=32, so A stride is still D_MODEL=64 but we only use 32 cols.
        // Better: write to A[tid][0..31] and set A[tid][32..63] = 0 (or just leave them).
        // The WMMA template with K_DIM=32 will only read the first 32 columns.
        #pragma unroll
        for (int kk = 0; kk < BLOCK_FF; kk++) {
            smem->A[tid][kk] = __float2bfloat16(up_tile[kk]);
        }
        // Zero out remaining columns 32..63 so WMMA doesn't read garbage
        // (K_DIM=32 means K_TILES=2, reading cols 0..15 and 16..31, so 32..63 not accessed)
        __syncthreads();

        // Load ffn_down weight tile to B (reinterpreted as [32][64]):
        // ffn_down is [D_FF, D_MODEL], rows [k..k+32], all 64 columns
        // B area: smem->B is [64][32] = 4096 bytes, but we need [32][64] = 4096 bytes = same size!
        // Reinterpret B pointer as __nv_bfloat16* for flat addressing.
        {
            __nv_bfloat16* B_flat = &smem->B[0][0];
            constexpr int TOTAL_ELEMS = D_HEAD * D_MODEL;  // 32*64=2048
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / SEQ;  // 16
            int base = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                int idx = base + i;
                int row = idx / D_MODEL;  // 0..31
                int col = idx % D_MODEL;  // 0..63
                B_flat[row * D_MODEL + col] = ffn_down[(k + row) * D_MODEL + col];
            }
        }
        __syncthreads();

        // WMMA: (128,32) x (32,64) -> (128,64)
        // A is [128][D_MODEL] with K_DIM=32 (only first 32 cols used), A_STRIDE=D_MODEL
        // B is [32][64] at smem->B reinterpreted, B_STRIDE=D_MODEL
        // C output is 128x64 f32 = 32KB. Use C1+C2 as contiguous 128x64 buffer.
        // C1 is at &smem->C1[0][0], C2 immediately follows. Together they are 128*64 floats.
        // But wait -- C1 is [128][32] and C2 is [128][32]. They are NOT contiguous as [128][64]
        // because C1[0] is row 0 cols 0-31, C1[1] is row 1 cols 0-31, etc.
        // We need C as [128][64] row-major = 128 rows of 64 floats each.
        // C1[128][32] + C2[128][32] is interleaved, not contiguous rows.
        //
        // Solution: use C1 as a flat buffer reinterpreted as [128][64].
        // C1 starts at offset 20480 in SharedMem and has 16KB.
        // C2 starts at offset 36864 and has 16KB.
        // Together that's 32KB = 128*64*4 bytes, but they're not contiguous
        // because there may be padding. Let's check:
        //   &smem->C1[0][0] is at offset 20480
        //   &smem->C2[0][0] is at offset 20480 + 128*32*4 = 20480 + 16384 = 36864
        // So C1 and C2 ARE contiguous! C1+C2 = 128*64 floats starting at C1.
        // We can interpret this as float[128][64] with stride 64.
        // But C1 is declared as [128][32], so smem->C1[i][j] accesses element
        // at offset (i*32+j)*4 from C1 start. For a [128][64] layout we need
        // offset (i*64+j)*4. These don't match!
        //
        // So we need to use flat pointer arithmetic.
        {
            float* C_wide = &smem->C1[0][0];  // treat C1+C2 as flat 128*64 buffer
            __nv_bfloat16* B_flat = &smem->B[0][0];  // [32][64] with stride 64=D_MODEL
            wmma_matmul_128xKxN<D_HEAD, D_MODEL, D_MODEL, D_MODEL>(
                &smem->A[0][0], B_flat, C_wide, warp_id);
        }
        __syncthreads();

        // Read down projection result from C_wide[128][64] and accumulate
        {
            float* C_wide = &smem->C1[0][0];
            #pragma unroll
            for (int d = 0; d < D_MODEL; d++) {
                ffn_down_accum[d] += C_wide[tid * D_MODEL + d];
            }
        }

        // Down projection perturbation: h_dot_bfd += sum(up_tile * b_fd_tile)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_FF; kk++) {
            h_dot_bfd += up_tile[kk] * vec[OFF_FD + k + kk];
        }

        // Restore h_norm2 to A for the next FFN tile iteration
        // (A was overwritten with up_tile for the down projection WMMA)
        #pragma unroll
        for (int d = 0; d < D_MODEL; d++) {
            smem->A[tid][d] = __float2bfloat16(h_norm2[d]);
        }
        __syncthreads();
    }

    // Apply FFN residual: h += ffn_down_accum + pert + bias
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h[d] += ffn_down_accum[d]
              + sign_sigma * h_dot_bfd * a_fd[d]
              + load_bf16_as_f32(ffn_down_bias, d)
              + sign_sigma * vec[OFF_FDB + d];
    }

    // ════════════════════════════════════════════════════════════════════
    // Phase 6: Final LayerNorm
    // ════════════════════════════════════════════════════════════════════
    float mean_f = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) mean_f += h[d];
    mean_f /= D_MODEL;

    float h_final[D_MODEL];
    float var_f = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h_final[d] = h[d] - mean_f;
        var_f += h_final[d] * h_final[d];
    }
    var_f /= D_MODEL;
    float inv_std_f = rsqrtf(var_f + 1e-5f);

    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        float s = load_bf16_as_f32(ln_final_scale, d) + sign_sigma * vec[OFF_LNFS + d];
        float b = load_bf16_as_f32(ln_final_bias, d) + sign_sigma * vec[OFF_LNFB + d];
        h_final[d] = s * h_final[d] * inv_std_f + b;
    }

    // ════════════════════════════════════════════════════════════════════
    // Phase 7: Output projection + label-smoothed CE loss
    // ════════════════════════════════════════════════════════════════════
    // Compute logits[VOCAB_PAD] = h_final @ output_proj + perturbation
    // output_proj is [D_MODEL, VOCAB_PAD] = [64, 128]
    // This is (128, 64) x (64, 128) -> (128, 128) = VOCAB_PAD logits per position.
    //
    // For WMMA: M=128, K=64, N=128. Output = 128*128*4 = 64KB. Too large for smem.
    // So we tile the output columns: compute (128,64)x(64,32)->(128,32) per tile,
    // 4 tiles to cover N=128. Each thread reads its 32 logit values per tile.

    // Precompute h_final . b_op for output projection perturbation
    float h_dot_bop = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h_dot_bop += h_final[d] * vec[OFF_OP + d];
    }

    // Write h_final to A as bf16 for WMMA
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        smem->A[tid][d] = __float2bfloat16(h_final[d]);
    }
    __syncthreads();

    const int label = y[actual_b * SEQ + tid];

    // Compute logits in tiles of 32, finding max and accumulating
    float max_logit = -FLT_MAX;
    float logits[VOCAB_PAD];

    for (int v_start = 0; v_start < VOCAB_PAD; v_start += D_HEAD) {
        // Load output_proj weight tile: [D_MODEL][D_HEAD] = [64][32]
        // output_proj[:, v_start..v_start+32]
        {
            constexpr int TOTAL_ELEMS = D_MODEL * D_HEAD;
            constexpr int ELEMS_PER_THREAD = TOTAL_ELEMS / SEQ;
            int base = tid * ELEMS_PER_THREAD;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; i++) {
                int idx = base + i;
                int row = idx / D_HEAD;
                int col = idx % D_HEAD;
                smem->B[row][col] = output_proj[row * VOCAB_PAD + v_start + col];
            }
        }
        __syncthreads();

        // WMMA: h_final(128,64) @ output_proj_tile(64,32) -> C1(128,32)
        wmma_matmul_128xKxN<D_MODEL, D_HEAD, D_MODEL, D_HEAD>(
            &smem->A[0][0], &smem->B[0][0], &smem->C1[0][0], warp_id);
        __syncthreads();

        // Read logits from C1, add perturbation
        #pragma unroll
        for (int j = 0; j < D_HEAD; j++) {
            int v = v_start + j;
            float val = smem->C1[tid][j];
            if (v < VOCAB) {
                float a_op_v = vec[OFF_OP + D_MODEL + v];
                val += sign_sigma * h_dot_bop * a_op_v;
            } else {
                val = -1e9f;
            }
            logits[v] = val / temperature;
            if (logits[v] > max_logit) max_logit = logits[v];
        }
        __syncthreads();
    }

    // Pass 2: compute sum_exp and log-partition
    float sum_exp = 0.0f;
    for (int v = 0; v < VOCAB_PAD; v++) {
        sum_exp += expf(logits[v] - max_logit);
    }
    float log_Z = max_logit + logf(sum_exp);

    // Label-smoothed CE: loss = -sum_v smooth[v] * log_softmax[v]
    float log_sm_label = logits[label] - log_Z;
    float sum_log_sm = 0.0f;
    for (int v = 0; v < VOCAB; v++) {
        sum_log_sm += (logits[v] - log_Z);
    }
    float ce = -((1.0f - alpha) * log_sm_label + (alpha / VOCAB) * sum_log_sm);

    // ════════════════════════════════════════════════════════════════════
    // CE reduction: sum across 128 threads to get total_ce = sum(ce) / SEQ
    // ════════════════════════════════════════════════════════════════════
    smem->reduce[tid] = ce;
    __syncthreads();

    // Tree reduction in shared memory
    for (int stride = SEQ / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem->reduce[tid] += smem->reduce[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the final result
    if (tid == 0) {
        float total_ce = smem->reduce[0] / SEQ;
        float* out_ptr = (pid_sign == 0) ? ce_pos : ce_neg;
        out_ptr[pid_p * batch + actual_b] = total_ce;
    }
}


// ─── XLA custom call interface ───
struct KernelParams {
    int32_t half_pop;
    int32_t batch;
    float sigma;
    float alpha;
    float temperature;
};

extern "C" void fused_transformer_ce_cuda(
    cudaStream_t stream,
    void** buffers,
    const char* opaque,
    size_t opaque_len,
    void* status  // XlaCustomCallStatus* -- ignored (success by default)
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
    // Scalars from opaque struct (not from buffers - avoids scalar buffer issues)
    const float sigma       = params->sigma;
    const float alpha_val   = params->alpha;
    const float temperature = params->temperature;

    // Output buffers (after 20 inputs: 17 weights + vecs + x + y)
    float* ce_pos = reinterpret_cast<float*>(buffers[20]);
    float* ce_neg = reinterpret_cast<float*>(buffers[21]);

    // Request extended shared memory (>48KB default)
    static bool smem_configured = false;
    if (!smem_configured) {
        cudaFuncSetAttribute(
            fused_transformer_ce_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sizeof(SharedMem)
        );
        smem_configured = true;
    }

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
