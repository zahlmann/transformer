"""
Dual-number fused transformer forward pass + CE loss kernel for EGGROLL.

Forward-mode AD: computes both primal CE loss and its exact directional derivative
along a rank-1 perturbation direction. No sigma, no antithetic pairs.

Interleaved primal + tangent: each operation computes (y, dy) from (x, dx).
Grid: (N_DIRS, BATCH). Output: (ce_primal, ce_tangent) each (N_DIRS, BATCH).

Tangent rules used:
  matmul:    d(X@W) = dX@W + X@dW, where dW = b⊗a (rank-1)
  layernorm: d(s*x_hat+b) = ds*x_hat + s*dx_hat + db
  softmax:   dp = p*(ds - sum(p*ds))
  gelu:      d(gelu(x)) = dx*(sig + 1.702*x*sig*(1-sig))
  CE:        d(-sum(t*log_sm)) = sum((softmax-t)*d_scaled)
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

OFF_AK  = tl.constexpr(0)
OFF_AO  = tl.constexpr(128)
OFF_AQ  = tl.constexpr(256)
OFF_AV  = tl.constexpr(384)
OFF_FD  = tl.constexpr(512)
OFF_FDB = tl.constexpr(832)
OFF_FU  = tl.constexpr(896)
OFF_FUB = tl.constexpr(1216)
OFF_LN1B = tl.constexpr(1472)
OFF_LN1S = tl.constexpr(1536)
OFF_LN2B = tl.constexpr(1600)
OFF_LN2S = tl.constexpr(1664)
OFF_LNFB = tl.constexpr(1728)
OFF_LNFS = tl.constexpr(1792)
OFF_OP  = tl.constexpr(1856)
OFF_PE  = tl.constexpr(1985)
OFF_TE  = tl.constexpr(2177)

_VEC_DIM  = tl.constexpr(2306)
_SEQ      = tl.constexpr(128)
_D_MODEL  = tl.constexpr(64)
_D_HEAD   = tl.constexpr(32)
_D_FF     = tl.constexpr(256)
_VOCAB    = tl.constexpr(65)
_VOCAB_PAD = tl.constexpr(128)
_BLOCK_K  = tl.constexpr(32)


@triton.jit
def _fused_dual_kernel(
    token_emb_ptr, pos_emb_ptr,
    ln1_scale_ptr, ln1_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    ln2_scale_ptr, ln2_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr,
    ffn_down_ptr, ffn_down_bias_ptr,
    ln_final_scale_ptr, ln_final_bias_ptr,
    output_proj_ptr,
    vecs_ptr,
    x_ptr, y_ptr,
    alpha_ptr, temperature_ptr,
    ce_primal_ptr, ce_tangent_ptr,
    N_DIRS: tl.constexpr,
    BATCH: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    alpha = tl.load(alpha_ptr).to(tl.float32)
    temperature = tl.load(temperature_ptr).to(tl.float32)
    vb = pid_d * _VEC_DIM
    offs_pos = tl.arange(0, _SEQ)
    offs_d = tl.arange(0, _D_MODEL)

    # ══════════════════════════════════════════════════════════
    # 1. EMBEDDING  (primal: h, tangent: dh)
    # ══════════════════════════════════════════════════════════
    x_seq = tl.load(x_ptr + pid_b * _SEQ + offs_pos)

    # Primal
    emb = tl.load(token_emb_ptr + x_seq[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)
    pos = tl.load(pos_emb_ptr + offs_pos[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)
    h = emb + pos

    # Tangent: dh = d(token_emb)[x] + d(pos_emb) where d(W) = b⊗a
    b_te = tl.load(vecs_ptr + vb + OFF_TE + x_seq)
    a_te = tl.load(vecs_ptr + vb + OFF_TE + _VOCAB + offs_d)
    b_pe = tl.load(vecs_ptr + vb + OFF_PE + offs_pos)
    a_pe = tl.load(vecs_ptr + vb + OFF_PE + _SEQ + offs_d)
    dh = b_te[:, None] * a_te[None, :] + b_pe[:, None] * a_pe[None, :]

    # ══════════════════════════════════════════════════════════
    # 2. LAYER NORM 1  (h,dh) -> (h_norm, dh_norm)
    # ══════════════════════════════════════════════════════════
    ln1_s = tl.load(ln1_scale_ptr + offs_d).to(tl.float32)
    ln1_b = tl.load(ln1_bias_ptr + offs_d).to(tl.float32)

    mean1 = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc1 = h - mean1
    var1 = tl.sum(hc1 * hc1, axis=1)[:, None] / _D_MODEL
    rsqrt1 = tl.math.rsqrt(var1 + 1e-5)
    x_hat1 = hc1 * rsqrt1
    h_norm = ln1_s[None, :] * x_hat1 + ln1_b[None, :]

    d_ln1_s = tl.load(vecs_ptr + vb + OFF_LN1S + offs_d)
    d_ln1_b = tl.load(vecs_ptr + vb + OFF_LN1B + offs_d)
    dmean1 = tl.sum(dh, axis=1)[:, None] / _D_MODEL
    dhc1 = dh - dmean1
    dot1 = tl.sum(x_hat1 * dhc1, axis=1)[:, None] / _D_MODEL
    dx_hat1 = rsqrt1 * (dhc1 - x_hat1 * dot1)
    dh_norm = d_ln1_s[None, :] * x_hat1 + ln1_s[None, :] * dx_hat1 + d_ln1_b[None, :]

    # ══════════════════════════════════════════════════════════
    # 3. MULTI-HEAD ATTENTION
    # ══════════════════════════════════════════════════════════
    scale = 0.17677669529663689  # 1/sqrt(32)
    b_q = tl.load(vecs_ptr + vb + OFF_AQ + offs_d)
    b_k = tl.load(vecs_ptr + vb + OFF_AK + offs_d)
    b_v = tl.load(vecs_ptr + vb + OFF_AV + offs_d)
    a_o = tl.load(vecs_ptr + vb + OFF_AO + _D_MODEL + offs_d)

    # Precompute h_norm . b_* (primal) for rank-1 weight tangent
    hn_bq = tl.sum(h_norm * b_q[None, :], axis=1)
    hn_bk = tl.sum(h_norm * b_k[None, :], axis=1)
    hn_bv = tl.sum(h_norm * b_v[None, :], axis=1)

    d_o_accum = tl.zeros((_SEQ,), dtype=tl.float32)

    for head in tl.range(2):
        offs_head = head * _D_HEAD + tl.arange(0, _D_HEAD)

        # Q primal: Q = h_norm @ Wq_head
        wq_h = tl.load(wq_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm.to(tl.bfloat16), wq_h).to(tl.float32)
        # Q tangent: dQ = dh_norm @ Wq + (h_norm.b_q) * a_q_head
        a_q_h = tl.load(vecs_ptr + vb + OFF_AQ + _D_MODEL + offs_head)
        dQ = tl.dot(dh_norm.to(tl.bfloat16), wq_h).to(tl.float32) + hn_bq[:, None] * a_q_h[None, :]

        # K primal + tangent
        wk_h = tl.load(wk_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        K = tl.dot(h_norm.to(tl.bfloat16), wk_h).to(tl.float32)
        a_k_h = tl.load(vecs_ptr + vb + OFF_AK + _D_MODEL + offs_head)
        dK = tl.dot(dh_norm.to(tl.bfloat16), wk_h).to(tl.float32) + hn_bk[:, None] * a_k_h[None, :]

        # V primal + tangent
        wv_h = tl.load(wv_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        V = tl.dot(h_norm.to(tl.bfloat16), wv_h).to(tl.float32)
        a_v_h = tl.load(vecs_ptr + vb + OFF_AV + _D_MODEL + offs_head)
        dV = tl.dot(dh_norm.to(tl.bfloat16), wv_h).to(tl.float32) + hn_bv[:, None] * a_v_h[None, :]

        # Attention scores primal + tangent
        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        causal = offs_pos[:, None] >= offs_pos[None, :]
        scores = tl.where(causal, scores, -1e9)

        d_scores = tl.dot(dQ.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        d_scores += tl.dot(Q.to(tl.bfloat16), tl.trans(dK.to(tl.bfloat16))).to(tl.float32) * scale
        d_scores = tl.where(causal, d_scores, 0.0)

        # Softmax primal
        max_s = tl.max(scores, axis=1)
        exp_s = tl.exp(scores - max_s[:, None])
        sum_s = tl.sum(exp_s, axis=1)
        attn_w = exp_s / sum_s[:, None]

        # Softmax tangent: d_attn = attn_w * (d_scores - sum(attn_w * d_scores))
        weighted_ds = tl.sum(attn_w * d_scores, axis=1)
        d_attn_w = attn_w * (d_scores - weighted_ds[:, None])

        # Attention output primal + tangent
        attn_out = tl.dot(attn_w.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)
        d_attn_out = tl.dot(d_attn_w.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)
        d_attn_out += tl.dot(attn_w.to(tl.bfloat16), dV.to(tl.bfloat16)).to(tl.float32)

        # O projection primal: h += attn_out @ Wo_head
        wo_h = tl.load(wo_ptr + offs_head[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        h += tl.dot(attn_out.to(tl.bfloat16), wo_h).to(tl.float32)
        # O projection tangent: dh += d_attn_out @ Wo_head
        dh += tl.dot(d_attn_out.to(tl.bfloat16), wo_h).to(tl.float32)

        # O weight rank-1 tangent: accumulate (attn_out . b_o_head) across heads
        b_o_h = tl.load(vecs_ptr + vb + OFF_AO + offs_head)
        d_o_accum += tl.sum(attn_out * b_o_h[None, :], axis=1)

    # Add O weight rank-1 tangent
    dh += d_o_accum[:, None] * a_o[None, :]

    # ══════════════════════════════════════════════════════════
    # 4. LAYER NORM 2
    # ══════════════════════════════════════════════════════════
    ln2_s = tl.load(ln2_scale_ptr + offs_d).to(tl.float32)
    ln2_b = tl.load(ln2_bias_ptr + offs_d).to(tl.float32)
    mean2 = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc2 = h - mean2
    var2 = tl.sum(hc2 * hc2, axis=1)[:, None] / _D_MODEL
    rsqrt2 = tl.math.rsqrt(var2 + 1e-5)
    x_hat2 = hc2 * rsqrt2
    h_norm2 = ln2_s[None, :] * x_hat2 + ln2_b[None, :]

    d_ln2_s = tl.load(vecs_ptr + vb + OFF_LN2S + offs_d)
    d_ln2_b = tl.load(vecs_ptr + vb + OFF_LN2B + offs_d)
    dmean2 = tl.sum(dh, axis=1)[:, None] / _D_MODEL
    dhc2 = dh - dmean2
    dot2 = tl.sum(x_hat2 * dhc2, axis=1)[:, None] / _D_MODEL
    dx_hat2 = rsqrt2 * (dhc2 - x_hat2 * dot2)
    dh_norm2 = d_ln2_s[None, :] * x_hat2 + ln2_s[None, :] * dx_hat2 + d_ln2_b[None, :]

    # ══════════════════════════════════════════════════════════
    # 5. FFN (K-tiled)
    # ══════════════════════════════════════════════════════════
    b_fu = tl.load(vecs_ptr + vb + OFF_FU + offs_d)
    hn2_bfu = tl.sum(h_norm2 * b_fu[None, :], axis=1)  # (SEQ,) primal dot

    ffn_down_accum = tl.zeros((_SEQ, _D_MODEL), dtype=tl.float32)
    d_ffn_down_accum = tl.zeros((_SEQ, _D_MODEL), dtype=tl.float32)
    fd_dot_accum = tl.zeros((_SEQ,), dtype=tl.float32)  # for FFN down rank-1 weight tangent
    a_fd = tl.load(vecs_ptr + vb + OFF_FD + _D_FF + offs_d)

    for k in tl.range(0, _D_FF, _BLOCK_K):
        offs_k = k + tl.arange(0, _BLOCK_K)

        # FFN up primal: pre_act = h_norm2 @ Wu_tile + bias
        wu_tile = tl.load(ffn_up_ptr + offs_d[:, None] * _D_FF + offs_k[None, :]).to(tl.bfloat16)
        pre_act = tl.dot(h_norm2.to(tl.bfloat16), wu_tile).to(tl.float32)
        bias_k = tl.load(ffn_up_bias_ptr + offs_k).to(tl.float32)
        pre_act = pre_act + bias_k[None, :]

        # FFN up tangent: d_pre_act = dh_norm2 @ Wu + (h_norm2.b_fu)*a_fu + v_bias
        a_fu_k = tl.load(vecs_ptr + vb + OFF_FU + _D_MODEL + offs_k)
        v_bias_k = tl.load(vecs_ptr + vb + OFF_FUB + offs_k)
        d_pre_act = tl.dot(dh_norm2.to(tl.bfloat16), wu_tile).to(tl.float32)
        d_pre_act = d_pre_act + hn2_bfu[:, None] * a_fu_k[None, :] + v_bias_k[None, :]

        # GELU primal
        sig = tl.sigmoid(1.702 * pre_act)
        act = pre_act * sig

        # GELU tangent
        d_act = d_pre_act * (sig + 1.702 * pre_act * sig * (1.0 - sig))

        # FFN down primal: accum += act @ Wd_tile
        wd_tile = tl.load(ffn_down_ptr + offs_k[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        ffn_down_accum += tl.dot(act.to(tl.bfloat16), wd_tile).to(tl.float32)

        # FFN down tangent: d_accum += d_act @ Wd_tile
        d_ffn_down_accum += tl.dot(d_act.to(tl.bfloat16), wd_tile).to(tl.float32)

        # FFN down weight rank-1 tangent: accumulate (act . b_fd_k)
        b_fd_k = tl.load(vecs_ptr + vb + OFF_FD + offs_k)
        fd_dot_accum += tl.sum(act * b_fd_k[None, :], axis=1)

    # Primal residual
    bias_down = tl.load(ffn_down_bias_ptr + offs_d).to(tl.float32)
    h = h + ffn_down_accum + bias_down[None, :]

    # Tangent residual: dh += d_ffn_down + fd_weight_tangent + d_bias_down
    v_bias_down = tl.load(vecs_ptr + vb + OFF_FDB + offs_d)
    dh = dh + d_ffn_down_accum + fd_dot_accum[:, None] * a_fd[None, :] + v_bias_down[None, :]

    # ══════════════════════════════════════════════════════════
    # 6. FINAL LAYER NORM
    # ══════════════════════════════════════════════════════════
    lnf_s = tl.load(ln_final_scale_ptr + offs_d).to(tl.float32)
    lnf_b = tl.load(ln_final_bias_ptr + offs_d).to(tl.float32)
    meanf = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hcf = h - meanf
    varf = tl.sum(hcf * hcf, axis=1)[:, None] / _D_MODEL
    rsqrtf = tl.math.rsqrt(varf + 1e-5)
    x_hatf = hcf * rsqrtf
    h_final = lnf_s[None, :] * x_hatf + lnf_b[None, :]

    d_lnf_s = tl.load(vecs_ptr + vb + OFF_LNFS + offs_d)
    d_lnf_b = tl.load(vecs_ptr + vb + OFF_LNFB + offs_d)
    dmeanf = tl.sum(dh, axis=1)[:, None] / _D_MODEL
    dhcf = dh - dmeanf
    dotf = tl.sum(x_hatf * dhcf, axis=1)[:, None] / _D_MODEL
    dx_hatf = rsqrtf * (dhcf - x_hatf * dotf)
    dh_final = d_lnf_s[None, :] * x_hatf + lnf_s[None, :] * dx_hatf + d_lnf_b[None, :]

    # ══════════════════════════════════════════════════════════
    # 7. OUTPUT PROJECTION + CE LOSS
    # ══════════════════════════════════════════════════════════
    offs_v = tl.arange(0, _VOCAB_PAD)
    op_w = tl.load(output_proj_ptr + offs_d[:, None] * _VOCAB_PAD + offs_v[None, :]).to(tl.bfloat16)

    # Primal logits
    logits = tl.dot(h_final.to(tl.bfloat16), op_w).to(tl.float32)
    logits = tl.where(offs_v[None, :] < _VOCAB, logits, -1e9)

    # Tangent logits: d_logits = dh_final @ W_out + (h_final . b_op) * a_op
    d_logits = tl.dot(dh_final.to(tl.bfloat16), op_w).to(tl.float32)
    b_op = tl.load(vecs_ptr + vb + OFF_OP + offs_d)
    a_op = tl.load(vecs_ptr + vb + OFF_OP + _D_MODEL + offs_v, mask=offs_v < _VOCAB, other=0.0)
    hf_bop = tl.sum(h_final * b_op[None, :], axis=1)
    d_logits = d_logits + hf_bop[:, None] * a_op[None, :]
    d_logits = tl.where(offs_v[None, :] < _VOCAB, d_logits, 0.0)

    # Primal CE with label smoothing
    y_labels = tl.load(y_ptr + pid_b * _SEQ + offs_pos)
    scaled = logits / temperature
    max_val = tl.max(scaled, axis=1)[:, None]
    exp_s = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_s, axis=1)[:, None]
    p = exp_s / sum_exp  # softmax
    log_sm = scaled - max_val - tl.log(sum_exp)

    one_hot = (offs_v[None, :] == y_labels[:, None]).to(tl.float32)
    smooth = (1.0 - alpha) * one_hot + alpha / _VOCAB
    smooth = tl.where(offs_v[None, :] < _VOCAB, smooth, 0.0)

    ce = -tl.sum(log_sm * smooth, axis=1)
    total_ce = tl.sum(ce) / _SEQ

    # Tangent CE: d_ce = sum((softmax - smooth) * d_scaled)
    d_scaled = d_logits / temperature
    d_ce = tl.sum((p - smooth) * d_scaled, axis=1)
    d_total_ce = tl.sum(d_ce) / _SEQ

    tl.store(ce_primal_ptr + pid_d * BATCH + pid_b, total_ce)
    tl.store(ce_tangent_ptr + pid_d * BATCH + pid_b, d_total_ce)


def fused_dual_forward(
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y, alpha, temperature,
):
    """Launch dual kernel. Returns (ce_primal, ce_tangent) each (N_DIRS, BATCH)."""
    N_DIRS = vecs.shape[0]
    BATCH = x.shape[0]
    VOCAB_PAD = 128
    output_proj_padded = jnp.pad(output_proj, [(0, 0), (0, VOCAB_PAD - output_proj.shape[1])])

    bf = lambda t: t.astype(jnp.bfloat16)
    return jt.triton_call(
        bf(token_emb), bf(pos_emb),
        bf(ln1_scale), bf(ln1_bias),
        bf(wq), bf(wk), bf(wv), bf(wo),
        bf(ln2_scale), bf(ln2_bias),
        bf(ffn_up), bf(ffn_up_bias), bf(ffn_down), bf(ffn_down_bias),
        bf(ln_final_scale), bf(ln_final_bias),
        bf(output_proj_padded),
        vecs,
        x.astype(jnp.int32), y.astype(jnp.int32),
        jnp.float32(alpha), jnp.float32(temperature),
        kernel=_fused_dual_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((N_DIRS, BATCH), jnp.float32),
            jax.ShapeDtypeStruct((N_DIRS, BATCH), jnp.float32),
        ],
        grid=(N_DIRS, BATCH),
        N_DIRS=N_DIRS,
        BATCH=BATCH,
        num_warps=4,
        num_stages=1,
    )
