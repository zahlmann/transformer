"""
Fused transformer decode kernel — one token through the whole model in one kernel call.

Reads KV cache from previous steps, computes attention over all past tokens,
outputs logits + new K/V vectors. Uses element-wise ops (not tl.dot) because
M=1 can't use tensor cores.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

D_MODEL    = tl.constexpr(64)
D_HEAD     = tl.constexpr(32)
D_FF       = tl.constexpr(256)
MAX_SEQ    = tl.constexpr(128)
BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)  # tile size for output projection loop


@triton.jit
def _decode_kernel(
    # Weights (bf16)
    token_emb_ptr, pos_emb_ptr,
    ln1_scale_ptr, ln1_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    ln2_scale_ptr, ln2_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr, ffn_down_ptr, ffn_down_bias_ptr,
    ln_final_scale_ptr, ln_final_bias_ptr,
    output_proj_ptr,
    # Input
    token_id_ptr, pos_ptr,
    k_cache_ptr, v_cache_ptr,
    # Outputs — full updated caches
    logits_ptr, k_cache_out_ptr, v_cache_out_ptr,
    # Constexpr parameters for variable vocab
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    d = tl.arange(0, D_MODEL)
    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    # ── Embedding ──
    h = (tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
       + tl.load(pos_emb_ptr + pos * D_MODEL + d).to(tl.float32))

    # ── Layer Norm 1 ──
    ln1_s = tl.load(ln1_scale_ptr + d).to(tl.float32)
    ln1_b = tl.load(ln1_bias_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_norm = ln1_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln1_b

    # ── Multi-head Attention with KV Cache ──
    scale = 0.17677669529663689  # 1/sqrt(32)
    seq = tl.arange(0, MAX_SEQ)
    mask = seq <= pos  # attend to positions 0..pos inclusive
    dh = tl.arange(0, D_HEAD)
    attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)

    for head in tl.range(2):
        hd = head * D_HEAD + dh
        cache_off = head * MAX_SEQ * D_HEAD

        # Q/K/V projections (element-wise vec @ mat since M=1)
        Q = tl.sum(h_norm[:, None] * tl.load(wq_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        K_new = tl.sum(h_norm[:, None] * tl.load(wk_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        V_new = tl.sum(h_norm[:, None] * tl.load(wv_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)

        # Load K cache, insert new K at current position
        K = tl.load(k_cache_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        K = tl.where(seq[:, None] == pos, K_new[None, :], K)
        # Write full updated K cache
        tl.store(k_cache_out_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], K.to(tl.bfloat16), mask=mask[:, None])
        tl.store(k_cache_out_ptr + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))

        # Attention scores → softmax → weighted V sum
        scores = tl.sum(Q[None, :] * K, axis=1) * scale
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores))
        attn_w = exp_s / tl.sum(exp_s)

        V = tl.load(v_cache_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        V = tl.where(seq[:, None] == pos, V_new[None, :], V)
        # Write full updated V cache
        tl.store(v_cache_out_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], V.to(tl.bfloat16), mask=mask[:, None])
        tl.store(v_cache_out_ptr + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

        attn_out = tl.sum(attn_w[:, None] * V, axis=0)

        # O projection
        attn_accum += tl.sum(attn_out[:, None] * tl.load(wo_ptr + hd[:, None] * D_MODEL + d[None, :]).to(tl.float32), axis=0)

    h = h + attn_accum

    # ── Layer Norm 2 ──
    ln2_s = tl.load(ln2_scale_ptr + d).to(tl.float32)
    ln2_b = tl.load(ln2_bias_ptr + d).to(tl.float32)
    mean2 = tl.sum(h) / D_MODEL
    hc2 = h - mean2
    h_norm2 = ln2_s * hc2 * tl.math.rsqrt(tl.sum(hc2 * hc2) / D_MODEL + 1e-5) + ln2_b

    # ── FFN (tiled) ──
    ffn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        up = tl.sum(h_norm2[:, None] * tl.load(ffn_up_ptr + d[:, None] * D_FF + kk[None, :]).to(tl.float32), axis=0)
        up += tl.load(ffn_up_bias_ptr + kk).to(tl.float32)
        act = up * tl.sigmoid(1.702 * up)
        ffn_accum += tl.sum(act[:, None] * tl.load(ffn_down_ptr + kk[:, None] * D_MODEL + d[None, :]).to(tl.float32), axis=0)
    h = h + ffn_accum + tl.load(ffn_down_bias_ptr + d).to(tl.float32)

    # ── Final Layer Norm ──
    lnf_s = tl.load(ln_final_scale_ptr + d).to(tl.float32)
    lnf_b = tl.load(ln_final_bias_ptr + d).to(tl.float32)
    mean_f = tl.sum(h) / D_MODEL
    hcf = h - mean_f
    h_final = lnf_s * hcf * tl.math.rsqrt(tl.sum(hcf * hcf) / D_MODEL + 1e-5) + lnf_b

    # ── Output Projection (tiled over vocab dimension) ──
    for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
        vv = v_start + tl.arange(0, VOCAB_TILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.float32)
        tile_logits = tl.sum(h_final[:, None] * out_w, axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)


def prepare_decode_weights_small(params, vocab_size):
    """Precompute bf16 weights + padded output proj once."""
    vocab_pad = ((vocab_size + 127) // 128) * 128
    pad_v = vocab_pad - params["output_proj"].shape[1]
    w = {k: v.astype(jnp.bfloat16) for k, v in params.items()}
    w["_output_proj_padded"] = jnp.pad(params["output_proj"], [(0, 0), (0, pad_v)]).astype(jnp.bfloat16)
    w["_vocab_pad"] = vocab_pad
    return w


def fused_decode(w, token_id, pos, k_cache, v_cache, vocab_size=65):
    """Run one decode step in a single fused kernel call.

    Args:
        w: precomputed bf16 weights from prepare_decode_weights_small()
        token_id: scalar int32, the new token
        pos: scalar int32, position index (0-based)
        k_cache: (2, 128, 32) bf16 — valid at positions 0..pos-1
        v_cache: (2, 128, 32) bf16
        vocab_size: actual vocabulary size

    Returns:
        logits:  (vocab_size,) float32
        k_cache: (2, 128, 32) bf16 — now valid at 0..pos
        v_cache: (2, 128, 32) bf16
    """
    vocab_pad = w["_vocab_pad"]

    logits_pad, k_out, v_out = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["layer0.ln1.scale"], w["layer0.ln1.bias"],
        w["layer0.attn.q"], w["layer0.attn.k"],
        w["layer0.attn.v"], w["layer0.attn.o"],
        w["layer0.ln2.scale"], w["layer0.ln2.bias"],
        w["layer0.ffn.up"], w["layer0.ffn.up_bias"],
        w["layer0.ffn.down"], w["layer0.ffn.down_bias"],
        w["ln_final.scale"], w["ln_final.bias"],
        w["_output_proj_padded"],
        jnp.int32(token_id),
        jnp.int32(pos),
        k_cache, v_cache,
        kernel=_decode_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct(k_cache.shape, jnp.bfloat16),
            jax.ShapeDtypeStruct(v_cache.shape, jnp.bfloat16),
        ],
        grid=(1,),
        num_warps=4,
        num_stages=1,
        VOCAB_SIZE=vocab_size,
        VOCAB_PAD=vocab_pad,
    )
    return logits_pad[:vocab_size], k_out, v_out
