"""
Dual-number fused transformer forward pass + CE loss kernel for EGGROLL.

Forward-mode AD: computes both primal CE loss and its exact directional derivative
along a rank-1 perturbation direction. No sigma, no antithetic pairs.

Two-phase execution per thread block:
  Phase A: Primal forward pass, stores key intermediates to shared memory (bf16)
  Phase B: Tangent propagation using stored intermediates + perturbation vectors

Grid: (N_DIRS, BATCH) — one block per (perturbation direction, batch element).
Output: directional derivative of CE loss per (direction, batch_element).
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

# Fixed perturbation vector offsets (alphabetical key order).
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
    alpha_ptr, temperature_ptr,
    # Output
    ce_primal_ptr, ce_tangent_ptr,
    # Grid constants
    N_DIRS: tl.constexpr,
    BATCH: tl.constexpr,
):
    pid_d = tl.program_id(0)  # perturbation direction index
    pid_b = tl.program_id(1)  # batch index

    alpha = tl.load(alpha_ptr).to(tl.float32)
    temperature = tl.load(temperature_ptr).to(tl.float32)

    vb = pid_d * _VEC_DIM
    offs_pos = tl.arange(0, _SEQ)
    offs_d = tl.arange(0, _D_MODEL)

    # ═══════════════════════════════════════════════════════════
    # PHASE A: Primal forward pass (no perturbation)
    # ═══════════════════════════════════════════════════════════

    # ── A1: Embedding -> h (128, 64) f32 ──
    x_seq = tl.load(x_ptr + pid_b * _SEQ + offs_pos)
    emb = tl.load(token_emb_ptr + x_seq[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)
    pos = tl.load(pos_emb_ptr + offs_pos[:, None] * _D_MODEL + offs_d[None, :]).to(tl.float32)
    h = emb + pos

    # ── A2: Layer Norm 1 -> h_norm ──
    ln1_s = tl.load(ln1_scale_ptr + offs_d).to(tl.float32)
    ln1_b = tl.load(ln1_bias_ptr + offs_d).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc = h - mean
    var = tl.sum(hc * hc, axis=1)[:, None] / _D_MODEL
    rsqrt_var = tl.math.rsqrt(var + 1e-5)
    x_hat1 = hc * rsqrt_var
    h_norm = ln1_s[None, :] * x_hat1 + ln1_b[None, :]

    # ── A3: Multi-head attention ──
    scale = 0.17677669529663689  # 1/sqrt(32)

    for head in tl.range(2):
        offs_head = head * _D_HEAD + tl.arange(0, _D_HEAD)

        wk_h = tl.load(wk_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        K = tl.dot(h_norm.to(tl.bfloat16), wk_h).to(tl.float32)

        wv_h = tl.load(wv_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        V = tl.dot(h_norm.to(tl.bfloat16), wv_h).to(tl.float32)

        wq_h = tl.load(wq_ptr + offs_d[:, None] * _D_MODEL + offs_head[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm.to(tl.bfloat16), wq_h).to(tl.float32)

        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        causal_mask = offs_pos[:, None] >= offs_pos[None, :]
        scores = tl.where(causal_mask, scores, -1e9)
        max_s = tl.max(scores, axis=1)
        scores_exp = tl.exp(scores - max_s[:, None])
        sum_s = tl.sum(scores_exp, axis=1)
        attn_w = scores_exp / sum_s[:, None]

        attn_out = tl.dot(attn_w.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)

        wo_h = tl.load(wo_ptr + offs_head[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        h += tl.dot(attn_out.to(tl.bfloat16), wo_h).to(tl.float32)

    # ── A4: Layer Norm 2 ──
    ln2_s = tl.load(ln2_scale_ptr + offs_d).to(tl.float32)
    ln2_b = tl.load(ln2_bias_ptr + offs_d).to(tl.float32)
    mean2 = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hc2 = h - mean2
    var2 = tl.sum(hc2 * hc2, axis=1)[:, None] / _D_MODEL
    rsqrt_var2 = tl.math.rsqrt(var2 + 1e-5)
    x_hat2 = hc2 * rsqrt_var2
    h_norm2 = ln2_s[None, :] * x_hat2 + ln2_b[None, :]

    # ── A5: FFN (K-tiled) ──
    ffn_down_accum = tl.zeros((_SEQ, _D_MODEL), dtype=tl.float32)

    for k in tl.range(0, _D_FF, _BLOCK_K):
        offs_k = k + tl.arange(0, _BLOCK_K)
        wu_tile = tl.load(ffn_up_ptr + offs_d[:, None] * _D_FF + offs_k[None, :]).to(tl.bfloat16)
        up_base = tl.dot(h_norm2.to(tl.bfloat16), wu_tile).to(tl.float32)
        bias_k = tl.load(ffn_up_bias_ptr + offs_k).to(tl.float32)
        pre_act = up_base + bias_k[None, :]
        ffn_up_act = pre_act * tl.sigmoid(1.702 * pre_act)

        wd_tile = tl.load(ffn_down_ptr + offs_k[:, None] * _D_MODEL + offs_d[None, :]).to(tl.bfloat16)
        ffn_down_accum += tl.dot(ffn_up_act.to(tl.bfloat16), wd_tile).to(tl.float32)

    bias_down = tl.load(ffn_down_bias_ptr + offs_d).to(tl.float32)
    h = h + ffn_down_accum + bias_down[None, :]

    # ── A6: Final Layer Norm ──
    lnf_s = tl.load(ln_final_scale_ptr + offs_d).to(tl.float32)
    lnf_b = tl.load(ln_final_bias_ptr + offs_d).to(tl.float32)
    mean_f = tl.sum(h, axis=1)[:, None] / _D_MODEL
    hcf = h - mean_f
    var_f = tl.sum(hcf * hcf, axis=1)[:, None] / _D_MODEL
    rsqrt_varf = tl.math.rsqrt(var_f + 1e-5)
    x_hatf = hcf * rsqrt_varf
    h_final = lnf_s[None, :] * x_hatf + lnf_b[None, :]

    # ── A7: Output projection + CE loss ──
    offs_v = tl.arange(0, _VOCAB_PAD)
    op_w = tl.load(output_proj_ptr + offs_d[:, None] * _VOCAB_PAD + offs_v[None, :]).to(tl.bfloat16)
    logits = tl.dot(h_final.to(tl.bfloat16), op_w).to(tl.float32)
    logits = tl.where(offs_v[None, :] < _VOCAB, logits, -1e9)

    y_labels = tl.load(y_ptr + pid_b * _SEQ + offs_pos)
    scaled = logits / temperature
    max_val = tl.max(scaled, axis=1)[:, None]
    exp_scaled = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_scaled, axis=1)[:, None]
    softmax_out = exp_scaled / sum_exp
    log_sm = scaled - max_val - tl.log(sum_exp)

    one_hot = (offs_v[None, :] == y_labels[:, None]).to(tl.float32)
    smooth = (1.0 - alpha) * one_hot + alpha / _VOCAB
    smooth = tl.where(offs_v[None, :] < _VOCAB, smooth, 0.0)

    ce = -tl.sum(log_sm * smooth, axis=1)
    total_ce = tl.sum(ce) / _SEQ

    # Store primal CE
    tl.store(ce_primal_ptr + pid_d * BATCH + pid_b, total_ce)

    # ═══════════════════════════════════════════════════════════
    # PHASE B: Tangent propagation (forward-mode AD)
    # ═══════════════════════════════════════════════════════════
    # TODO: implement tangent propagation
    # For now, output 0.0 as placeholder
    tl.store(ce_tangent_ptr + pid_d * BATCH + pid_b, 0.0)


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
