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
// Plus a reduction buffer for CE summation
struct SharedMem {
    float K[SEQ][D_HEAD];    // 128*32 = 4096 floats = 16KB
    float V[SEQ][D_HEAD];    // 128*32 = 4096 floats = 16KB
    float reduce[SEQ];       // 128 floats = 512B (for CE reduction)
    // Total: ~32.5KB — well within 100KB limit
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

/**
 * Main fused transformer + CE loss kernel.
 *
 * Thread mapping: 128 threads (4 warps). Thread tid owns sequence position tid.
 * Each thread computes one row of all (128, 64) or (128, 32) matrices.
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

    // Perturbation vector base pointer for this member
    const float* vec = vecs + pid_p * VEC_DIM;

    // Shared memory
    extern __shared__ char smem_raw[];
    SharedMem* smem = reinterpret_cast<SharedMem*>(smem_raw);

    // Batch element index
    const int actual_b = pid_bt * BATCH_TILE;

    // ════════════════════════════════════════════════════════════════════
    // Phase 1: Embedding + perturbation → h[D_MODEL]
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
    // Phase 2: LayerNorm 1 → h_norm[D_MODEL]
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
    // Phase 3: Multi-head attention (FlashAttention via shared memory)
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

    for (int head = 0; head < N_HEADS; head++) {
        const int head_off = head * D_HEAD;

        // ── Compute K[tid][D_HEAD] and V[tid][D_HEAD], store to shared memory ──
        // K = h_norm @ wk_head + sign_sigma * h_dot_bk * a_k_head
        // V = h_norm @ wv_head + sign_sigma * h_dot_bv * a_v_head
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            // K: matmul row tid of h_norm with column (head_off+dh) of wk
            // wk is [D_MODEL, D_MODEL], column index = head_off+dh
            float k_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_MODEL; d++) {
                k_val += h_norm[d] * load_bf16_as_f32(wk, d * D_MODEL + head_off + dh);
            }
            float a_k_dh = vec[OFF_AK + D_MODEL + head_off + dh];
            k_val += sign_sigma * h_dot_bk * a_k_dh;
            smem->K[tid][dh] = k_val;

            // V: same pattern
            float v_val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_MODEL; d++) {
                v_val += h_norm[d] * load_bf16_as_f32(wv, d * D_MODEL + head_off + dh);
            }
            float a_v_dh = vec[OFF_AV + D_MODEL + head_off + dh];
            v_val += sign_sigma * h_dot_bv * a_v_dh;
            smem->V[tid][dh] = v_val;
        }

        __syncthreads();

        // ── Compute Q[D_HEAD] for this thread's position ──
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
        // For this thread (query position tid), compute attention over all
        // key positions 0..127, tiled in blocks of BLOCK_KV=32.
        float m_running = -FLT_MAX;  // running max of scores
        float l_running = 0.0f;       // running sum of exp(score - m)
        float o_acc[D_HEAD];           // running attention output accumulator
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) o_acc[dh] = 0.0f;

        for (int kv_start = 0; kv_start < SEQ; kv_start += BLOCK_KV) {
            // Compute dot products Q . K^T for this tile of KV positions
            float scores[BLOCK_KV];
            float tile_max = -FLT_MAX;

            #pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                int kv_pos = kv_start + j;
                // Causal mask: position tid can only attend to positions <= tid
                if (kv_pos > tid) {
                    scores[j] = -FLT_MAX;
                } else {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int dh = 0; dh < D_HEAD; dh++) {
                        dot += Q[dh] * smem->K[kv_pos][dh];
                    }
                    scores[j] = dot * scale;
                }
                if (scores[j] > tile_max) tile_max = scores[j];
            }

            // Online softmax update
            float m_new = fmaxf(m_running, tile_max);
            float correction = expf(m_running - m_new);

            // Rescale existing accumulator
            l_running *= correction;
            #pragma unroll
            for (int dh = 0; dh < D_HEAD; dh++) {
                o_acc[dh] *= correction;
            }

            // Accumulate this tile
            #pragma unroll
            for (int j = 0; j < BLOCK_KV; j++) {
                float p = expf(scores[j] - m_new);
                l_running += p;
                int kv_pos = kv_start + j;
                #pragma unroll
                for (int dh = 0; dh < D_HEAD; dh++) {
                    o_acc[dh] += p * smem->V[kv_pos][dh];
                }
            }

            m_running = m_new;
        }

        // Finalize: divide by sum of weights
        float inv_l = 1.0f / l_running;
        float attn_out[D_HEAD];
        #pragma unroll
        for (int dh = 0; dh < D_HEAD; dh++) {
            attn_out[dh] = o_acc[dh] * inv_l;
        }

        // ── O projection: h += attn_out @ wo_head ──
        // wo is [D_MODEL, D_MODEL], but for this head we read rows [head_off..head_off+D_HEAD)
        // wo_head is [D_HEAD, D_MODEL]  (rows head_off..head_off+31, all D_MODEL columns)
        // output[d] = sum_dh attn_out[dh] * wo[head_off+dh][d]
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
        // o_pert += sum(attn_out * b_o_head)
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
    // Phase 5: FFN (tiled over D_FF in blocks of BLOCK_FF=32)
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

    for (int k = 0; k < D_FF; k += BLOCK_FF) {
        // Up projection: up_tile[BLOCK_FF] = h_norm2 @ ffn_up_tile + perturbation + bias
        float up_tile[BLOCK_FF];
        #pragma unroll
        for (int kk = 0; kk < BLOCK_FF; kk++) {
            int ff_idx = k + kk;
            // h_norm2[D_MODEL] @ ffn_up[:, ff_idx]
            // ffn_up is [D_MODEL, D_FF], column ff_idx
            float val = 0.0f;
            #pragma unroll
            for (int d = 0; d < D_MODEL; d++) {
                val += h_norm2[d] * load_bf16_as_f32(ffn_up, d * D_FF + ff_idx);
            }
            // Perturbation: sign_sigma * hn2_dot_bfu * a_fu[ff_idx]
            float a_fu_k = vec[OFF_FU + D_MODEL + ff_idx];
            val += sign_sigma * hn2_dot_bfu * a_fu_k;
            // Bias + bias perturbation
            val += load_bf16_as_f32(ffn_up_bias, ff_idx);
            val += sign_sigma * vec[OFF_FUB + ff_idx];
            // GELU activation
            up_tile[kk] = gelu_approx(val);
        }

        // Down projection: ffn_down_accum[D_MODEL] += up_tile @ ffn_down_tile
        // ffn_down is [D_FF, D_MODEL], rows k..k+BLOCK_FF
        #pragma unroll
        for (int d = 0; d < D_MODEL; d++) {
            float val = 0.0f;
            #pragma unroll
            for (int kk = 0; kk < BLOCK_FF; kk++) {
                val += up_tile[kk] * load_bf16_as_f32(ffn_down, (k + kk) * D_MODEL + d);
            }
            ffn_down_accum[d] += val;
        }

        // Down projection perturbation: h_dot_bfd += sum(up_tile * b_fd_tile)
        #pragma unroll
        for (int kk = 0; kk < BLOCK_FF; kk++) {
            h_dot_bfd += up_tile[kk] * vec[OFF_FD + k + kk];
        }
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
    // output_proj is [D_MODEL, VOCAB_PAD]

    // Precompute h_final . b_op for output projection perturbation
    float h_dot_bop = 0.0f;
    #pragma unroll
    for (int d = 0; d < D_MODEL; d++) {
        h_dot_bop += h_final[d] * vec[OFF_OP + d];
    }

    // Compute logits, apply perturbation, and compute CE loss in a streaming
    // fashion over vocab to avoid materializing all VOCAB_PAD logits.
    // We need two passes: first to find max for numerical stability, then for softmax.
    // But we can use online softmax (single pass) for CE.

    // Get the label for this position
    const int label = y[actual_b * SEQ + tid];

    // Two-pass CE: need full log-softmax for label smoothing.
    // Pass 1: compute all logits, find max
    float max_logit = -FLT_MAX;
    float logits[VOCAB_PAD];
    for (int v = 0; v < VOCAB_PAD; v++) {
        float val = 0.0f;
        #pragma unroll
        for (int d = 0; d < D_MODEL; d++) {
            val += h_final[d] * load_bf16_as_f32(output_proj, d * VOCAB_PAD + v);
        }
        // Perturbation (only for valid vocab entries)
        if (v < VOCAB) {
            float a_op_v = vec[OFF_OP + D_MODEL + v];
            val += sign_sigma * h_dot_bop * a_op_v;
        } else {
            val = -1e9f;  // mask padded entries
        }
        logits[v] = val / temperature;
        if (logits[v] > max_logit) max_logit = logits[v];
    }

    // Pass 2: compute sum_exp and log-partition
    float sum_exp = 0.0f;
    for (int v = 0; v < VOCAB_PAD; v++) {
        sum_exp += expf(logits[v] - max_logit);
    }
    float log_Z = max_logit + logf(sum_exp);

    // Label-smoothed CE: loss = -sum_v smooth[v] * log_softmax[v]
    // smooth[v] = (1-alpha)*one_hot[v==label] + alpha/VOCAB  (for v < VOCAB)
    // = -[(1-alpha)*log_softmax[label] + (alpha/VOCAB)*sum_v log_softmax[v]]
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
    float sigma;
    float alpha;
    float temperature;
};

extern "C" void fused_transformer_ce_cuda(
    cudaStream_t stream,
    void** buffers,
    const char* opaque,
    size_t opaque_len,
    void* status  // XlaCustomCallStatus* — ignored (success by default)
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
