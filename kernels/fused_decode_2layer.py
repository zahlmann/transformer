"""
Fully fused 2-layer decode kernel: embedding → layer0 → layer1 → output in ONE launch.

The per-layer block_decode approach used 3 kernel launches per decode step
(layer0 + layer1 + output). Python/JAX dispatch overhead per jt.triton_call
is ~1-3ms, so 3 launches × 63 steps = ~200-600ms of pure overhead.

This kernel fuses everything into a single launch. With M=1 (single token decode),
register pressure is trivial (~35KB peak) regardless of d_model.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)


@triton.jit
def _fused_decode_2layer(
    # Embedding weights
    token_emb_ptr, pos_emb_ptr,
    # Layer 0 weights
    l0_ln1_s_ptr, l0_ln1_b_ptr,
    l0_wq_ptr, l0_wk_ptr, l0_wv_ptr, l0_wo_ptr,
    l0_ln2_s_ptr, l0_ln2_b_ptr,
    l0_up_ptr, l0_up_b_ptr, l0_down_ptr, l0_down_b_ptr,
    # Layer 1 weights
    l1_ln1_s_ptr, l1_ln1_b_ptr,
    l1_wq_ptr, l1_wk_ptr, l1_wv_ptr, l1_wo_ptr,
    l1_ln2_s_ptr, l1_ln2_b_ptr,
    l1_up_ptr, l1_up_b_ptr, l1_down_ptr, l1_down_b_ptr,
    # Final LN + output
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    # Decode inputs
    token_id_ptr, pos_ptr,
    l0_kc_ptr, l0_vc_ptr,
    l1_kc_ptr, l1_vc_ptr,
    # Outputs
    logits_ptr,
    l0_k_new_ptr, l0_v_new_ptr,
    l1_k_new_ptr, l1_v_new_ptr,
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    d = tl.arange(0, D_MODEL)
    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    # ── Embedding ──
    h = (tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
       + tl.load(pos_emb_ptr + pos * D_MODEL + d).to(tl.float32))

    # ════════════════════════════════════════════
    # LAYER 0
    # ════════════════════════════════════════════

    # ── LN1 ──
    ln_s = tl.load(l0_ln1_s_ptr + d).to(tl.float32)
    ln_b = tl.load(l0_ln1_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

    # ── Attention ──
    scale = 0.17677669529663689  # 1/sqrt(32)
    seq = tl.arange(0, MAX_SEQ)
    mask = seq <= pos
    dh = tl.arange(0, D_HEAD)
    attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        cache_off = head * MAX_SEQ * D_HEAD

        Q = tl.sum(h_norm[:, None] * tl.load(l0_wq_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        K_new = tl.sum(h_norm[:, None] * tl.load(l0_wk_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        V_new = tl.sum(h_norm[:, None] * tl.load(l0_wv_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)

        tl.store(l0_k_new_ptr + head * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(l0_v_new_ptr + head * D_HEAD + dh, V_new.to(tl.bfloat16))

        K = tl.load(l0_kc_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        K = tl.where(seq[:, None] == pos, K_new[None, :], K)
        scores = tl.sum(Q[None, :] * K, axis=1) * scale
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores))
        attn_w = exp_s / tl.sum(exp_s)

        V = tl.load(l0_vc_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        V = tl.where(seq[:, None] == pos, V_new[None, :], V)
        attn_out = tl.sum(attn_w[:, None] * V, axis=0)
        attn_accum += tl.sum(attn_out[:, None] * tl.load(l0_wo_ptr + hd[:, None] * D_MODEL + d[None, :]).to(tl.float32), axis=0)

    h = h + attn_accum

    # ── LN2 + FFN ──
    ln_s = tl.load(l0_ln2_s_ptr + d).to(tl.float32)
    ln_b = tl.load(l0_ln2_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

    ffn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        up = tl.sum(h_norm[:, None] * tl.load(l0_up_ptr + d[:, None] * D_FF + kk[None, :]).to(tl.float32), axis=0)
        up += tl.load(l0_up_b_ptr + kk).to(tl.float32)
        act = up * tl.sigmoid(1.702 * up)
        ffn_accum += tl.sum(act[:, None] * tl.load(l0_down_ptr + kk[:, None] * D_MODEL + d[None, :]).to(tl.float32), axis=0)
    h = h + ffn_accum + tl.load(l0_down_b_ptr + d).to(tl.float32)

    # ════════════════════════════════════════════
    # LAYER 1
    # ════════════════════════════════════════════

    # ── LN1 ──
    ln_s = tl.load(l1_ln1_s_ptr + d).to(tl.float32)
    ln_b = tl.load(l1_ln1_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

    # ── Attention ──
    attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        cache_off = head * MAX_SEQ * D_HEAD

        Q = tl.sum(h_norm[:, None] * tl.load(l1_wq_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        K_new = tl.sum(h_norm[:, None] * tl.load(l1_wk_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        V_new = tl.sum(h_norm[:, None] * tl.load(l1_wv_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)

        tl.store(l1_k_new_ptr + head * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(l1_v_new_ptr + head * D_HEAD + dh, V_new.to(tl.bfloat16))

        K = tl.load(l1_kc_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        K = tl.where(seq[:, None] == pos, K_new[None, :], K)
        scores = tl.sum(Q[None, :] * K, axis=1) * scale
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores))
        attn_w = exp_s / tl.sum(exp_s)

        V = tl.load(l1_vc_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        V = tl.where(seq[:, None] == pos, V_new[None, :], V)
        attn_out = tl.sum(attn_w[:, None] * V, axis=0)
        attn_accum += tl.sum(attn_out[:, None] * tl.load(l1_wo_ptr + hd[:, None] * D_MODEL + d[None, :]).to(tl.float32), axis=0)

    h = h + attn_accum

    # ── LN2 + FFN ──
    ln_s = tl.load(l1_ln2_s_ptr + d).to(tl.float32)
    ln_b = tl.load(l1_ln2_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

    ffn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        up = tl.sum(h_norm[:, None] * tl.load(l1_up_ptr + d[:, None] * D_FF + kk[None, :]).to(tl.float32), axis=0)
        up += tl.load(l1_up_b_ptr + kk).to(tl.float32)
        act = up * tl.sigmoid(1.702 * up)
        ffn_accum += tl.sum(act[:, None] * tl.load(l1_down_ptr + kk[:, None] * D_MODEL + d[None, :]).to(tl.float32), axis=0)
    h = h + ffn_accum + tl.load(l1_down_b_ptr + d).to(tl.float32)

    # ════════════════════════════════════════════
    # OUTPUT
    # ════════════════════════════════════════════

    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_final = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

    for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
        vv = v_start + tl.arange(0, VOCAB_TILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.float32)
        tile_logits = tl.sum(h_final[:, None] * out_w, axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)


# ──────────────────────────────────────────────────────────────────────

def prepare_decode_weights(params, config, vocab_size):
    """Precompute bf16 weights + padded output proj once. Call before decode loop."""
    vocab_pad = ((vocab_size + 127) // 128) * 128
    pad_v = vocab_pad - vocab_size
    w = {k: v.astype(jnp.bfloat16) for k, v in params.items()}
    w["_output_proj_padded"] = jnp.pad(params["output_proj"], [(0, 0), (0, pad_v)]).astype(jnp.bfloat16)
    w["_vocab_pad"] = vocab_pad
    return w


def fused_decode_2layer(w, config, token_id, pos, k_caches, v_caches, vocab_size):
    """Fully fused 2-layer decode: one kernel call per token.

    Args:
        w: precomputed bf16 weights from prepare_decode_weights()

    Returns: logits, [k_cache_l0, k_cache_l1], [v_cache_l0, v_cache_l1]
    """
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["_vocab_pad"]

    logits_pad, l0_k_new, l0_v_new, l1_k_new, l1_v_new = jt.triton_call(
        # Embedding
        w["token_emb"], w["pos_emb"],
        # Layer 0
        w["layer0.ln1.scale"], w["layer0.ln1.bias"],
        w["layer0.attn.q"], w["layer0.attn.k"], w["layer0.attn.v"], w["layer0.attn.o"],
        w["layer0.ln2.scale"], w["layer0.ln2.bias"],
        w["layer0.ffn.up"], w["layer0.ffn.up_bias"], w["layer0.ffn.down"], w["layer0.ffn.down_bias"],
        # Layer 1
        w["layer1.ln1.scale"], w["layer1.ln1.bias"],
        w["layer1.attn.q"], w["layer1.attn.k"], w["layer1.attn.v"], w["layer1.attn.o"],
        w["layer1.ln2.scale"], w["layer1.ln2.bias"],
        w["layer1.ffn.up"], w["layer1.ffn.up_bias"], w["layer1.ffn.down"], w["layer1.ffn.down_bias"],
        # Final LN + output
        w["ln_final.scale"], w["ln_final.bias"],
        w["_output_proj_padded"],
        # Decode inputs
        jnp.int32(token_id), jnp.int32(pos),
        k_caches[0], v_caches[0],
        k_caches[1], v_caches[1],
        kernel=_fused_decode_2layer,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((n_heads, d_head), jnp.bfloat16),
            jax.ShapeDtypeStruct((n_heads, d_head), jnp.bfloat16),
            jax.ShapeDtypeStruct((n_heads, d_head), jnp.bfloat16),
            jax.ShapeDtypeStruct((n_heads, d_head), jnp.bfloat16),
        ],
        grid=(1,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, MAX_SEQ=max_seq,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
    )

    new_k = [k_caches[0].at[:, pos, :].set(l0_k_new), k_caches[1].at[:, pos, :].set(l1_k_new)]
    new_v = [v_caches[0].at[:, pos, :].set(l0_v_new), v_caches[1].at[:, pos, :].set(l1_v_new)]
    return logits_pad[:vocab_size], new_k, new_v
