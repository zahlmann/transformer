"""
Fused transformer forward pass + CE loss kernel for EGGROLL ES training.

Processes one perturbation x one batch sequence x one sign per thread block.
Grid: (HALF_POP, BATCH, 2) where dim 2 is +sigma/-sigma.

Architecture: d_model=64, n_heads=2, d_head=32, d_ff=256, vocab=65, seq=128.
Uses num_warps=8 (256 threads) for sufficient register budget.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

# Fixed perturbation vector offsets (alphabetical key order from build_param_spec).
# Must be tl.constexpr for access inside @triton.jit kernel.
OFF_AK  = tl.constexpr(0)     # layer0.attn.k (64,64): b(64)+a(64)
OFF_AO  = tl.constexpr(128)   # layer0.attn.o (64,64): b(64)+a(64)
OFF_AQ  = tl.constexpr(256)   # layer0.attn.q (64,64): b(64)+a(64)
OFF_AV  = tl.constexpr(384)   # layer0.attn.v (64,64): b(64)+a(64)
OFF_FD  = tl.constexpr(512)   # layer0.ffn.down (256,64): b(256)+a(64)
OFF_FDB = tl.constexpr(832)   # layer0.ffn.down_bias (64,)
OFF_FU  = tl.constexpr(896)   # layer0.ffn.up (64,256): b(64)+a(256)
OFF_FUB = tl.constexpr(1216)  # layer0.ffn.up_bias (256,)
OFF_LN1B = tl.constexpr(1472) # layer0.ln1.bias (64,)
OFF_LN1S = tl.constexpr(1536) # layer0.ln1.scale (64,)
OFF_LN2B = tl.constexpr(1600) # layer0.ln2.bias (64,)
OFF_LN2S = tl.constexpr(1664) # layer0.ln2.scale (64,)
OFF_LNFB = tl.constexpr(1728) # ln_final.bias (64,)
OFF_LNFS = tl.constexpr(1792) # ln_final.scale (64,)
OFF_OP  = tl.constexpr(1856)  # output_proj (64,65): b(64)+a(65)
OFF_PE  = tl.constexpr(1985)  # pos_emb (128,64): b(128)+a(64)
OFF_TE  = tl.constexpr(2177)  # token_emb (65,64): b(65)+a(64)

_VEC_DIM  = tl.constexpr(2306)
_SEQ      = tl.constexpr(128)
_D_MODEL  = tl.constexpr(64)
_D_HEAD   = tl.constexpr(32)
_D_FF     = tl.constexpr(256)
_VOCAB    = tl.constexpr(65)
_VOCAB_PAD = tl.constexpr(128)
_BLOCK_K  = tl.constexpr(32)


@triton.jit
def _fused_transformer_ce_kernel(
    # Base weights (bf16)
    token_emb_ptr, pos_emb_ptr,
    ln1_scale_ptr, ln1_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    ln2_scale_ptr, ln2_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr,
    ffn_down_ptr, ffn_down_bias_ptr,
    ln_final_scale_ptr, ln_final_bias_ptr,
    output_proj_ptr,
    # Perturbation vectors (f32)
    vecs_ptr,
    # Input data
    x_ptr, y_ptr,
    # Scalars
    sigma_ptr, alpha_ptr, temperature_ptr,
    # Output
    partial_ce_pos_ptr, partial_ce_neg_ptr,
    # Grid constants
    HALF_POP: tl.constexpr,
    BATCH: tl.constexpr,
):
    pid_p = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sign = tl.program_id(2)

    sigma_val = tl.load(sigma_ptr).to(tl.float32)
    sign_sigma = tl.where(pid_sign == 0, sigma_val, -sigma_val)
    alpha = tl.load(alpha_ptr).to(tl.float32)
    temperature = tl.load(temperature_ptr).to(tl.float32)

    vb = pid_p * _VEC_DIM
    offs_pos = tl.arange(0, _SEQ)
    offs_d = tl.arange(0, _D_MODEL)

    # ── Phase 1: Embedding + perturbation -> h (128, 64) f32 ──
    x_seq = tl.load(x_ptr + pid_b * _SEQ + offs_pos)

    emb = tl.load(token_emb_ptr + x_seq[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)
    pos = tl.load(pos_emb_ptr + offs_pos[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)

    b_te = tl.load(vecs_ptr + vb + OFF_TE + x_seq)
    a_te = tl.load(vecs_ptr + vb + OFF_TE + _VOCAB + offs_d)
    b_pe = tl.load(vecs_ptr + vb + OFF_PE + offs_pos)
    a_pe = tl.load(vecs_ptr + vb + OFF_PE + _SEQ + offs_d)

    h = emb + pos + sign_sigma * (
        b_te[:, None] * a_te[None, :] + b_pe[:, None] * a_pe[None, :]
    )

    # ── Phase 2: Layer Norm 1 -> h_norm ──
    ln1_s = tl.load(ln1_scale_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN1S + offs_d)
    ln1_b = tl.load(ln1_bias_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN1B + offs_d)

    mean = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc = h - mean
    var = tl.sum(hc * hc, axis=1)[:, None] / _D_MODEL
    h_norm = ln1_s[None, :] * hc * tl.math.rsqrt(var + 1e-5) + ln1_b[None, :]

    # ── Phase 3: Multi-head attention ──
    b_q = tl.load(vecs_ptr + vb + OFF_AQ + offs_d)
    b_k = tl.load(vecs_ptr + vb + OFF_AK + offs_d)
    b_v = tl.load(vecs_ptr + vb + OFF_AV + offs_d)
    h_dot_bq = tl.sum(h_norm * b_q[None, :], axis=1)
    h_dot_bk = tl.sum(h_norm * b_k[None, :], axis=1)
    h_dot_bv = tl.sum(h_norm * b_v[None, :], axis=1)

    a_o = tl.load(vecs_ptr + vb + OFF_AO + _D_MODEL + offs_d)
    o_pert = tl.zeros((_SEQ,), dtype=tl.float32)
    scale = 0.17677669529663689  # 1/sqrt(32) = 1/sqrt(D_HEAD)

    for head in tl.static_range(2):
        offs_head = head * _D_HEAD + tl.arange(0, _D_HEAD)

        # K
        wk_h = tl.load(wk_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        K = tl.dot(h_norm.to(tl.bfloat16), wk_h).to(tl.float32)
        a_k_h = tl.load(vecs_ptr + vb + OFF_AK + _D_MODEL + offs_head)
        K = K + sign_sigma * h_dot_bk[:, None] * a_k_h[None, :]

        # V
        wv_h = tl.load(wv_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        V = tl.dot(h_norm.to(tl.bfloat16), wv_h).to(tl.float32)
        a_v_h = tl.load(vecs_ptr + vb + OFF_AV + _D_MODEL + offs_head)
        V = V + sign_sigma * h_dot_bv[:, None] * a_v_h[None, :]

        # Q
        wq_h = tl.load(wq_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm.to(tl.bfloat16), wq_h).to(tl.float32)
        a_q_h = tl.load(vecs_ptr + vb + OFF_AQ + _D_MODEL + offs_head)
        Q = Q + sign_sigma * h_dot_bq[:, None] * a_q_h[None, :]

        # Attention scores (128, 128)
        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        causal_mask = offs_pos[:, None] >= offs_pos[None, :]
        scores = tl.where(causal_mask, scores, -1e9)
        max_s = tl.max(scores, axis=1)
        scores = tl.exp(scores - max_s[:, None])
        sum_s = tl.sum(scores, axis=1)
        attn_w = scores / sum_s[:, None]

        # Attention output
        attn_out = tl.dot(attn_w.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)

        # O projection base
        wo_h = tl.load(wo_ptr + offs_head[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        h += tl.dot(attn_out.to(tl.bfloat16), wo_h).to(tl.float32)

        # O projection perturbation accumulation
        b_o_h = tl.load(vecs_ptr + vb + OFF_AO + offs_head)
        o_pert += tl.sum(attn_out * b_o_h[None, :], axis=1)

    h = h + sign_sigma * o_pert[:, None] * a_o[None, :]

    # ── Phase 4: Layer Norm 2 ──
    ln2_s = tl.load(ln2_scale_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN2S + offs_d)
    ln2_b = tl.load(ln2_bias_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN2B + offs_d)

    mean2 = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc2 = h - mean2
    var2 = tl.sum(hc2 * hc2, axis=1)[:, None] / _D_MODEL
    h_norm2 = ln2_s[None, :] * hc2 * tl.math.rsqrt(var2 + 1e-5) + ln2_b[None, :]

    # ── Phase 5: FFN (K-tiled) ──
    b_fu = tl.load(vecs_ptr + vb + OFF_FU + offs_d)
    hn2_dot_bfu = tl.sum(h_norm2 * b_fu[None, :], axis=1)

    ffn_down_accum = tl.zeros((_SEQ, _D_MODEL), dtype=tl.float32)
    h_dot_bfd = tl.zeros((_SEQ,), dtype=tl.float32)
    a_fd = tl.load(vecs_ptr + vb + OFF_FD + _D_FF + offs_d)

    for k in tl.static_range(0, _D_FF, _BLOCK_K):
        offs_k = k + tl.arange(0, _BLOCK_K)

        wu_tile = tl.load(ffn_up_ptr + offs_d[:, None] * _D_FF + offs_k[None, :]).to(tl.bfloat16)
        up_base = tl.dot(h_norm2.to(tl.bfloat16), wu_tile).to(tl.float32)

        a_fu_k = tl.load(vecs_ptr + vb + OFF_FU + _D_MODEL + offs_k)
        up_pert = sign_sigma * hn2_dot_bfu[:, None] * a_fu_k[None, :]

        bias_k = tl.load(ffn_up_bias_ptr + offs_k).to(tl.float32)
        v_bias_k = tl.load(vecs_ptr + vb + OFF_FUB + offs_k)

        pre_act = up_base + up_pert + bias_k[None, :] + sign_sigma * v_bias_k[None, :]
        ffn_up_act = pre_act * tl.sigmoid(1.702 * pre_act)

        wd_tile = tl.load(ffn_down_ptr + offs_k[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        ffn_down_accum += tl.dot(ffn_up_act.to(tl.bfloat16), wd_tile).to(tl.float32)

        b_fd_k = tl.load(vecs_ptr + vb + OFF_FD + offs_k)
        h_dot_bfd += tl.sum(ffn_up_act * b_fd_k[None, :], axis=1)

    bias_down = tl.load(ffn_down_bias_ptr + offs_d).to(tl.float32)
    v_bias_down = tl.load(vecs_ptr + vb + OFF_FDB + offs_d)
    h = h + ffn_down_accum + sign_sigma * h_dot_bfd[:, None] * a_fd[None, :] \
        + bias_down[None, :] + sign_sigma * v_bias_down[None, :]

    # ── Phase 6: Final Layer Norm ──
    lnf_s = tl.load(ln_final_scale_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LNFS + offs_d)
    lnf_b = tl.load(ln_final_bias_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LNFB + offs_d)

    mean_f = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hcf = h - mean_f
    var_f = tl.sum(hcf * hcf, axis=1)[:, None] / _D_MODEL
    h_final = lnf_s[None, :] * hcf * tl.math.rsqrt(var_f + 1e-5) + lnf_b[None, :]

    # ── Phase 7: Output projection + CE loss ──
    offs_v = tl.arange(0, _VOCAB_PAD)

    op_w = tl.load(output_proj_ptr + offs_d[:, None] * _VOCAB_PAD + offs_v[None, :]).to(tl.bfloat16)
    logits = tl.dot(h_final.to(tl.bfloat16), op_w).to(tl.float32)

    b_op = tl.load(vecs_ptr + vb + OFF_OP + offs_d)
    a_op = tl.load(vecs_ptr + vb + OFF_OP + _D_MODEL + offs_v, mask=offs_v < _VOCAB, other=0.0)
    h_dot_bop = tl.sum(h_final * b_op[None, :], axis=1)
    logits = logits + sign_sigma * h_dot_bop[:, None] * a_op[None, :]
    logits = tl.where(offs_v[None, :] < _VOCAB, logits, -1e9)

    # Label-smoothed CE
    y_labels = tl.load(y_ptr + pid_b * _SEQ + offs_pos)

    scaled = logits / temperature
    max_val = tl.max(scaled, axis=1)[:, None]
    log_sm = scaled - max_val - tl.log(tl.sum(tl.exp(scaled - max_val), axis=1)[:, None])

    one_hot = (offs_v[None, :] == y_labels[:, None]).to(tl.float32)
    smooth = (1.0 - alpha) * one_hot + alpha / _VOCAB
    smooth = tl.where(offs_v[None, :] < _VOCAB, smooth, 0.0)

    ce = -tl.sum(log_sm * smooth, axis=1)
    total_ce = tl.sum(ce) / _SEQ

    out_ptr = tl.where(pid_sign == 0, partial_ce_pos_ptr, partial_ce_neg_ptr)
    tl.store(out_ptr + pid_p * BATCH + pid_b, total_ce)


# ═══════════════════════════════════════════════════════════════════════════
# Split kernel variant: Kernel A (attention) + Kernel B (FFN + CE)
# ═══════════════════════════════════════════════════════════════════════════

@triton.jit
def _attn_kernel(
    # Base weights (bf16)
    token_emb_ptr, pos_emb_ptr,
    ln1_scale_ptr, ln1_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    # Perturbation vectors (f32)
    vecs_ptr,
    # Input data
    x_ptr,
    # Scalars
    sigma_ptr,
    # Output: scratch buffer for h_after_attn
    scratch_ptr,
    # Grid constants
    CHUNK: tl.constexpr,
    BATCH: tl.constexpr,
    P_OFFSET: tl.constexpr,
):
    """Kernel A: Embedding + LN1 + Multi-head Attention + O projection + residual.

    Writes h_after_attn (SEQ, D_MODEL) per work item to scratch buffer.
    Grid: (CHUNK, BATCH, 2) where dim 2 is +sigma/-sigma.
    """
    pid_p_local = tl.program_id(0)   # perturbation index within chunk
    pid_b = tl.program_id(1)         # batch index
    pid_sign = tl.program_id(2)      # 0 = +sigma, 1 = -sigma

    pid_p = pid_p_local + P_OFFSET   # global perturbation index

    sigma_val = tl.load(sigma_ptr).to(tl.float32)
    sign_sigma = tl.where(pid_sign == 0, sigma_val, -sigma_val)

    vb = pid_p * _VEC_DIM
    offs_pos = tl.arange(0, _SEQ)
    offs_d = tl.arange(0, _D_MODEL)

    # ── Phase 1: Embedding + perturbation -> h (128, 64) f32 ──
    x_seq = tl.load(x_ptr + pid_b * _SEQ + offs_pos)

    emb = tl.load(token_emb_ptr + x_seq[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)
    pos = tl.load(pos_emb_ptr + offs_pos[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)

    b_te = tl.load(vecs_ptr + vb + OFF_TE + x_seq)
    a_te = tl.load(vecs_ptr + vb + OFF_TE + _VOCAB + offs_d)
    b_pe = tl.load(vecs_ptr + vb + OFF_PE + offs_pos)
    a_pe = tl.load(vecs_ptr + vb + OFF_PE + _SEQ + offs_d)

    h = emb + pos + sign_sigma * (
        b_te[:, None] * a_te[None, :] + b_pe[:, None] * a_pe[None, :]
    )

    # ── Phase 2: Layer Norm 1 -> h_norm ──
    ln1_s = tl.load(ln1_scale_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN1S + offs_d)
    ln1_b = tl.load(ln1_bias_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN1B + offs_d)

    mean = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc = h - mean
    var = tl.sum(hc * hc, axis=1)[:, None] / _D_MODEL
    h_norm = ln1_s[None, :] * hc * tl.math.rsqrt(var + 1e-5) + ln1_b[None, :]

    # ── Phase 3: Multi-head attention ──
    b_q = tl.load(vecs_ptr + vb + OFF_AQ + offs_d)
    b_k = tl.load(vecs_ptr + vb + OFF_AK + offs_d)
    b_v = tl.load(vecs_ptr + vb + OFF_AV + offs_d)
    h_dot_bq = tl.sum(h_norm * b_q[None, :], axis=1)
    h_dot_bk = tl.sum(h_norm * b_k[None, :], axis=1)
    h_dot_bv = tl.sum(h_norm * b_v[None, :], axis=1)

    a_o = tl.load(vecs_ptr + vb + OFF_AO + _D_MODEL + offs_d)
    o_pert = tl.zeros((_SEQ,), dtype=tl.float32)
    scale = 0.17677669529663689  # 1/sqrt(32) = 1/sqrt(D_HEAD)

    for head in tl.static_range(2):
        offs_head = head * _D_HEAD + tl.arange(0, _D_HEAD)

        # K
        wk_h = tl.load(wk_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        K = tl.dot(h_norm.to(tl.bfloat16), wk_h).to(tl.float32)
        a_k_h = tl.load(vecs_ptr + vb + OFF_AK + _D_MODEL + offs_head)
        K = K + sign_sigma * h_dot_bk[:, None] * a_k_h[None, :]

        # V
        wv_h = tl.load(wv_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        V = tl.dot(h_norm.to(tl.bfloat16), wv_h).to(tl.float32)
        a_v_h = tl.load(vecs_ptr + vb + OFF_AV + _D_MODEL + offs_head)
        V = V + sign_sigma * h_dot_bv[:, None] * a_v_h[None, :]

        # Q
        wq_h = tl.load(wq_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm.to(tl.bfloat16), wq_h).to(tl.float32)
        a_q_h = tl.load(vecs_ptr + vb + OFF_AQ + _D_MODEL + offs_head)
        Q = Q + sign_sigma * h_dot_bq[:, None] * a_q_h[None, :]

        # Attention scores (128, 128)
        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        causal_mask = offs_pos[:, None] >= offs_pos[None, :]
        scores = tl.where(causal_mask, scores, -1e9)
        max_s = tl.max(scores, axis=1)
        scores = tl.exp(scores - max_s[:, None])
        sum_s = tl.sum(scores, axis=1)
        attn_w = scores / sum_s[:, None]

        # Attention output
        attn_out = tl.dot(attn_w.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)

        # O projection base
        wo_h = tl.load(wo_ptr + offs_head[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        h += tl.dot(attn_out.to(tl.bfloat16), wo_h).to(tl.float32)

        # O projection perturbation accumulation
        b_o_h = tl.load(vecs_ptr + vb + OFF_AO + offs_head)
        o_pert += tl.sum(attn_out * b_o_h[None, :], axis=1)

    h = h + sign_sigma * o_pert[:, None] * a_o[None, :]

    # ── Write h_after_attn to scratch buffer ──
    # scratch layout: (CHUNK * BATCH * 2, SEQ, D_MODEL)
    # slot index: pid_sign * CHUNK * BATCH + pid_p_local * BATCH + pid_b
    slot = pid_sign * CHUNK * BATCH + pid_p_local * BATCH + pid_b
    scratch_base = slot * _SEQ * _D_MODEL
    tl.store(
        scratch_ptr + scratch_base + offs_pos[:, None] * _D_MODEL + offs_d[None, :],
        h,
    )


@triton.jit
def _ffn_ce_kernel(
    # Base weights (bf16)
    ln2_scale_ptr, ln2_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr,
    ffn_down_ptr, ffn_down_bias_ptr,
    ln_final_scale_ptr, ln_final_bias_ptr,
    output_proj_ptr,
    # Perturbation vectors (f32)
    vecs_ptr,
    # Input data
    y_ptr,
    # Scalars
    sigma_ptr, alpha_ptr, temperature_ptr,
    # Input: scratch buffer with h_after_attn
    scratch_ptr,
    # Output
    partial_ce_pos_ptr, partial_ce_neg_ptr,
    # Grid constants
    CHUNK: tl.constexpr,
    BATCH: tl.constexpr,
    P_OFFSET: tl.constexpr,
):
    """Kernel B: LN2 + FFN + Final LN + Output projection + CE loss.

    Reads h_after_attn from scratch buffer, writes CE to output arrays.
    Grid: (CHUNK, BATCH, 2) where dim 2 is +sigma/-sigma.
    Output arrays have shape (CHUNK, BATCH) -- one row per local perturbation.
    """
    pid_p_local = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_sign = tl.program_id(2)

    pid_p = pid_p_local + P_OFFSET   # global perturbation index (for vecs lookup)

    sigma_val = tl.load(sigma_ptr).to(tl.float32)
    sign_sigma = tl.where(pid_sign == 0, sigma_val, -sigma_val)
    alpha = tl.load(alpha_ptr).to(tl.float32)
    temperature = tl.load(temperature_ptr).to(tl.float32)

    vb = pid_p * _VEC_DIM
    offs_pos = tl.arange(0, _SEQ)
    offs_d = tl.arange(0, _D_MODEL)

    # ── Read h_after_attn from scratch buffer ──
    slot = pid_sign * CHUNK * BATCH + pid_p_local * BATCH + pid_b
    scratch_base = slot * _SEQ * _D_MODEL
    h = tl.load(scratch_ptr + scratch_base + offs_pos[:, None] * _D_MODEL + offs_d[None, :])

    # ── Phase 4: Layer Norm 2 ──
    ln2_s = tl.load(ln2_scale_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN2S + offs_d)
    ln2_b = tl.load(ln2_bias_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LN2B + offs_d)

    mean2 = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc2 = h - mean2
    var2 = tl.sum(hc2 * hc2, axis=1)[:, None] / _D_MODEL
    h_norm2 = ln2_s[None, :] * hc2 * tl.math.rsqrt(var2 + 1e-5) + ln2_b[None, :]

    # ── Phase 5: FFN (K-tiled) ──
    b_fu = tl.load(vecs_ptr + vb + OFF_FU + offs_d)
    hn2_dot_bfu = tl.sum(h_norm2 * b_fu[None, :], axis=1)

    ffn_down_accum = tl.zeros((_SEQ, _D_MODEL), dtype=tl.float32)
    h_dot_bfd = tl.zeros((_SEQ,), dtype=tl.float32)
    a_fd = tl.load(vecs_ptr + vb + OFF_FD + _D_FF + offs_d)

    for k in tl.static_range(0, _D_FF, _BLOCK_K):
        offs_k = k + tl.arange(0, _BLOCK_K)

        wu_tile = tl.load(ffn_up_ptr + offs_d[:, None] * _D_FF + offs_k[None, :]).to(tl.bfloat16)
        up_base = tl.dot(h_norm2.to(tl.bfloat16), wu_tile).to(tl.float32)

        a_fu_k = tl.load(vecs_ptr + vb + OFF_FU + _D_MODEL + offs_k)
        up_pert = sign_sigma * hn2_dot_bfu[:, None] * a_fu_k[None, :]

        bias_k = tl.load(ffn_up_bias_ptr + offs_k).to(tl.float32)
        v_bias_k = tl.load(vecs_ptr + vb + OFF_FUB + offs_k)

        pre_act = up_base + up_pert + bias_k[None, :] + sign_sigma * v_bias_k[None, :]
        ffn_up_act = pre_act * tl.sigmoid(1.702 * pre_act)

        wd_tile = tl.load(ffn_down_ptr + offs_k[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        ffn_down_accum += tl.dot(ffn_up_act.to(tl.bfloat16), wd_tile).to(tl.float32)

        b_fd_k = tl.load(vecs_ptr + vb + OFF_FD + offs_k)
        h_dot_bfd += tl.sum(ffn_up_act * b_fd_k[None, :], axis=1)

    bias_down = tl.load(ffn_down_bias_ptr + offs_d).to(tl.float32)
    v_bias_down = tl.load(vecs_ptr + vb + OFF_FDB + offs_d)
    h = h + ffn_down_accum + sign_sigma * h_dot_bfd[:, None] * a_fd[None, :] \
        + bias_down[None, :] + sign_sigma * v_bias_down[None, :]

    # ── Phase 6: Final Layer Norm ──
    lnf_s = tl.load(ln_final_scale_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LNFS + offs_d)
    lnf_b = tl.load(ln_final_bias_ptr + offs_d).to(tl.float32) + sign_sigma * tl.load(vecs_ptr + vb + OFF_LNFB + offs_d)

    mean_f = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hcf = h - mean_f
    var_f = tl.sum(hcf * hcf, axis=1)[:, None] / _D_MODEL
    h_final = lnf_s[None, :] * hcf * tl.math.rsqrt(var_f + 1e-5) + lnf_b[None, :]

    # ── Phase 7: Output projection + CE loss ──
    offs_v = tl.arange(0, _VOCAB_PAD)

    op_w = tl.load(output_proj_ptr + offs_d[:, None] * _VOCAB_PAD + offs_v[None, :]).to(tl.bfloat16)
    logits = tl.dot(h_final.to(tl.bfloat16), op_w).to(tl.float32)

    b_op = tl.load(vecs_ptr + vb + OFF_OP + offs_d)
    a_op = tl.load(vecs_ptr + vb + OFF_OP + _D_MODEL + offs_v, mask=offs_v < _VOCAB, other=0.0)
    h_dot_bop = tl.sum(h_final * b_op[None, :], axis=1)
    logits = logits + sign_sigma * h_dot_bop[:, None] * a_op[None, :]
    logits = tl.where(offs_v[None, :] < _VOCAB, logits, -1e9)

    # Label-smoothed CE
    y_labels = tl.load(y_ptr + pid_b * _SEQ + offs_pos)

    scaled = logits / temperature
    max_val = tl.max(scaled, axis=1)[:, None]
    log_sm = scaled - max_val - tl.log(tl.sum(tl.exp(scaled - max_val), axis=1)[:, None])

    one_hot = (offs_v[None, :] == y_labels[:, None]).to(tl.float32)
    smooth = (1.0 - alpha) * one_hot + alpha / _VOCAB
    smooth = tl.where(offs_v[None, :] < _VOCAB, smooth, 0.0)

    ce = -tl.sum(log_sm * smooth, axis=1)
    total_ce = tl.sum(ce) / _SEQ

    out_ptr = tl.where(pid_sign == 0, partial_ce_pos_ptr, partial_ce_neg_ptr)
    tl.store(out_ptr + pid_p_local * BATCH + pid_b, total_ce)


def fused_transformer_ce_both_split(
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y, sigma, alpha, temperature,
    chunk_size=64,
):
    """Split-kernel variant: processes in chunks to limit scratch memory.

    Kernel A (attention) writes h_after_attn to scratch buffer.
    Kernel B (FFN+CE) reads scratch and produces CE loss.

    Returns (partial_ce_pos, partial_ce_neg) each of shape (HALF_POP, BATCH),
    identical to fused_transformer_ce_both.
    """
    HALF_POP = vecs.shape[0]
    BATCH = x.shape[0]
    SEQ = 128
    D_MODEL = 64
    VOCAB_PAD = 128

    output_proj_padded = jnp.pad(output_proj, [(0, 0), (0, VOCAB_PAD - output_proj.shape[1])])

    # Cast base weights to bf16
    token_emb_bf = token_emb.astype(jnp.bfloat16)
    pos_emb_bf = pos_emb.astype(jnp.bfloat16)
    ln1_scale_bf = ln1_scale.astype(jnp.bfloat16)
    ln1_bias_bf = ln1_bias.astype(jnp.bfloat16)
    wq_bf = wq.astype(jnp.bfloat16)
    wk_bf = wk.astype(jnp.bfloat16)
    wv_bf = wv.astype(jnp.bfloat16)
    wo_bf = wo.astype(jnp.bfloat16)
    ln2_scale_bf = ln2_scale.astype(jnp.bfloat16)
    ln2_bias_bf = ln2_bias.astype(jnp.bfloat16)
    ffn_up_bf = ffn_up.astype(jnp.bfloat16)
    ffn_up_bias_bf = ffn_up_bias.astype(jnp.bfloat16)
    ffn_down_bf = ffn_down.astype(jnp.bfloat16)
    ffn_down_bias_bf = ffn_down_bias.astype(jnp.bfloat16)
    ln_final_scale_bf = ln_final_scale.astype(jnp.bfloat16)
    ln_final_bias_bf = ln_final_bias.astype(jnp.bfloat16)
    output_proj_bf = output_proj_padded.astype(jnp.bfloat16)

    sigma_f32 = jnp.float32(sigma)
    alpha_f32 = jnp.float32(alpha)
    temperature_f32 = jnp.float32(temperature)
    x_i32 = x.astype(jnp.int32)
    y_i32 = y.astype(jnp.int32)

    # Process in chunks of perturbation members
    n_chunks = HALF_POP // chunk_size
    assert HALF_POP % chunk_size == 0, f"HALF_POP ({HALF_POP}) must be divisible by chunk_size ({chunk_size})"

    # Scratch buffer: (chunk_size * BATCH * 2, SEQ, D_MODEL) in f32
    # Memory: chunk_size * BATCH * 2 * 128 * 64 * 4 bytes
    scratch_elems = chunk_size * BATCH * 2 * SEQ * D_MODEL

    ce_pos_chunks = []
    ce_neg_chunks = []

    for c in range(n_chunks):
        p_offset = c * chunk_size

        # Kernel A: attention -> scratch
        (scratch,) = jt.triton_call(
            token_emb_bf, pos_emb_bf,
            ln1_scale_bf, ln1_bias_bf,
            wq_bf, wk_bf, wv_bf, wo_bf,
            vecs,
            x_i32,
            sigma_f32,
            kernel=_attn_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((scratch_elems,), jnp.float32),
            ],
            grid=(chunk_size, BATCH, 2),
            CHUNK=chunk_size,
            BATCH=BATCH,
            P_OFFSET=p_offset,
            num_warps=4,
            num_stages=1,
        )

        # Kernel B: FFN + CE, reads scratch, writes (CHUNK, BATCH) outputs
        (ce_pos_chunk, ce_neg_chunk) = jt.triton_call(
            ln2_scale_bf, ln2_bias_bf,
            ffn_up_bf, ffn_up_bias_bf,
            ffn_down_bf, ffn_down_bias_bf,
            ln_final_scale_bf, ln_final_bias_bf,
            output_proj_bf,
            vecs,
            y_i32,
            sigma_f32, alpha_f32, temperature_f32,
            scratch,
            kernel=_ffn_ce_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((chunk_size, BATCH), jnp.float32),
                jax.ShapeDtypeStruct((chunk_size, BATCH), jnp.float32),
            ],
            grid=(chunk_size, BATCH, 2),
            CHUNK=chunk_size,
            BATCH=BATCH,
            P_OFFSET=p_offset,
            num_warps=4,
            num_stages=1,
        )

        ce_pos_chunks.append(ce_pos_chunk)
        ce_neg_chunks.append(ce_neg_chunk)

    # Concatenate chunks along perturbation axis
    partial_ce_pos = jnp.concatenate(ce_pos_chunks, axis=0)
    partial_ce_neg = jnp.concatenate(ce_neg_chunks, axis=0)
    return partial_ce_pos, partial_ce_neg


def fused_transformer_ce_both(
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y, sigma, alpha, temperature,
):
    """Launch fused kernel for all perturbations, both +sigma and -sigma.

    Returns (partial_ce_pos, partial_ce_neg) each of shape (HALF_POP, BATCH).
    Caller should sum over axis=1 and divide by BATCH for mean CE.
    """
    HALF_POP = vecs.shape[0]
    BATCH = x.shape[0]
    VOCAB_PAD = 128

    output_proj_padded = jnp.pad(output_proj, [(0, 0), (0, VOCAB_PAD - output_proj.shape[1])])

    token_emb_bf = token_emb.astype(jnp.bfloat16)
    pos_emb_bf = pos_emb.astype(jnp.bfloat16)
    ln1_scale_bf = ln1_scale.astype(jnp.bfloat16)
    ln1_bias_bf = ln1_bias.astype(jnp.bfloat16)
    wq_bf = wq.astype(jnp.bfloat16)
    wk_bf = wk.astype(jnp.bfloat16)
    wv_bf = wv.astype(jnp.bfloat16)
    wo_bf = wo.astype(jnp.bfloat16)
    ln2_scale_bf = ln2_scale.astype(jnp.bfloat16)
    ln2_bias_bf = ln2_bias.astype(jnp.bfloat16)
    ffn_up_bf = ffn_up.astype(jnp.bfloat16)
    ffn_up_bias_bf = ffn_up_bias.astype(jnp.bfloat16)
    ffn_down_bf = ffn_down.astype(jnp.bfloat16)
    ffn_down_bias_bf = ffn_down_bias.astype(jnp.bfloat16)
    ln_final_scale_bf = ln_final_scale.astype(jnp.bfloat16)
    ln_final_bias_bf = ln_final_bias.astype(jnp.bfloat16)
    output_proj_bf = output_proj_padded.astype(jnp.bfloat16)

    return jt.triton_call(
        token_emb_bf, pos_emb_bf,
        ln1_scale_bf, ln1_bias_bf,
        wq_bf, wk_bf, wv_bf, wo_bf,
        ln2_scale_bf, ln2_bias_bf,
        ffn_up_bf, ffn_up_bias_bf, ffn_down_bf, ffn_down_bias_bf,
        ln_final_scale_bf, ln_final_bias_bf,
        output_proj_bf,
        vecs,
        x.astype(jnp.int32), y.astype(jnp.int32),
        jnp.float32(sigma), jnp.float32(alpha), jnp.float32(temperature),
        kernel=_fused_transformer_ce_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((HALF_POP, BATCH), jnp.float32),
            jax.ShapeDtypeStruct((HALF_POP, BATCH), jnp.float32),
        ],
        grid=(HALF_POP, BATCH, 2),
        HALF_POP=HALF_POP,
        BATCH=BATCH,
        num_warps=4,
        num_stages=1,
    )
