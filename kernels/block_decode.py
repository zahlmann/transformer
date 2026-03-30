"""
Per-layer decode kernel for d_model >= 128, with JAX orchestration for multi-layer.

M=1 decode has trivial register pressure (h is just a 1D vector), so a single
fused kernel per layer works fine. JAX handles embedding, output projection,
and the layer loop.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)


@triton.jit
def _decode_layer_kernel(
    # h input (d_model,) f32
    h_ptr,
    # Layer weights
    ln1_scale_ptr, ln1_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    ln2_scale_ptr, ln2_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr, ffn_down_ptr, ffn_down_bias_ptr,
    # Decode input
    pos_ptr,
    k_cache_ptr, v_cache_ptr,
    # Outputs
    h_out_ptr, k_new_ptr, v_new_ptr,
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
):
    d = tl.arange(0, D_MODEL)
    pos = tl.load(pos_ptr)
    h = tl.load(h_ptr + d).to(tl.float32)

    # ── Layer Norm 1 ──
    ln1_s = tl.load(ln1_scale_ptr + d).to(tl.float32)
    ln1_b = tl.load(ln1_bias_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_norm = ln1_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln1_b

    # ── Multi-head Attention with KV Cache ──
    scale = 0.17677669529663689  # 1/sqrt(32)
    seq = tl.arange(0, MAX_SEQ)
    mask = seq <= pos
    dh = tl.arange(0, D_HEAD)
    attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        cache_off = head * MAX_SEQ * D_HEAD

        # Q/K/V projections (element-wise since M=1)
        Q = tl.sum(h_norm[:, None] * tl.load(wq_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        K_new = tl.sum(h_norm[:, None] * tl.load(wk_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)
        V_new = tl.sum(h_norm[:, None] * tl.load(wv_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.float32), axis=0)

        tl.store(k_new_ptr + head * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(v_new_ptr + head * D_HEAD + dh, V_new.to(tl.bfloat16))

        # Load K cache, insert new K
        K = tl.load(k_cache_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        K = tl.where(seq[:, None] == pos, K_new[None, :], K)

        # Attention scores → softmax
        scores = tl.sum(Q[None, :] * K, axis=1) * scale
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores))
        attn_w = exp_s / tl.sum(exp_s)

        V = tl.load(v_cache_ptr + cache_off + seq[:, None] * D_HEAD + dh[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        V = tl.where(seq[:, None] == pos, V_new[None, :], V)
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

    tl.store(h_out_ptr + d, h)


@triton.jit
def _decode_output_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    output_proj_ptr,
    logits_ptr,
    D_MODEL: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    """Final LN + tiled output projection for single-token decode."""
    d = tl.arange(0, D_MODEL)
    h = tl.load(h_ptr + d).to(tl.float32)

    ln_s = tl.load(ln_scale_ptr + d).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d).to(tl.float32)
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
# Python orchestrator
# ──────────────────────────────────────────────────────────────────────

def prepare_decode_weights_block(params, config, vocab_size):
    """Precompute bf16 weights + padded output proj once. Call before decode loop."""
    vocab_pad = ((vocab_size + 127) // 128) * 128
    pad_v = vocab_pad - vocab_size
    w = {k: v.astype(jnp.bfloat16) for k, v in params.items()}
    w["_output_proj_padded"] = jnp.pad(params["output_proj"], [(0, 0), (0, pad_v)]).astype(jnp.bfloat16)
    w["_vocab_pad"] = vocab_pad
    return w


def block_decode(w, config, token_id, pos, all_k_caches, all_v_caches, vocab_size):
    """Multi-layer decode: one token through all layers.

    Args:
        w: precomputed bf16 weights from prepare_decode_weights_block()
        config: model config
        token_id: scalar int32
        pos: scalar int32
        all_k_caches: list of (n_heads, max_seq, d_head) bf16 per layer
        all_v_caches: list of (n_heads, max_seq, d_head) bf16 per layer
        vocab_size: actual vocabulary size

    Returns:
        logits: (vocab_size,) float32
        all_k_caches: updated list
        all_v_caches: updated list
    """
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["_vocab_pad"]

    # Embedding (JAX)
    h = (w["token_emb"][token_id].astype(jnp.float32)
         + w["pos_emb"][pos].astype(jnp.float32))

    new_k_caches = []
    new_v_caches = []

    for layer in range(n_layers):
        p = f"layer{layer}"

        h_out, k_new, v_new = jt.triton_call(
            h,
            w[f"{p}.ln1.scale"], w[f"{p}.ln1.bias"],
            w[f"{p}.attn.q"], w[f"{p}.attn.k"],
            w[f"{p}.attn.v"], w[f"{p}.attn.o"],
            w[f"{p}.ln2.scale"], w[f"{p}.ln2.bias"],
            w[f"{p}.ffn.up"], w[f"{p}.ffn.up_bias"],
            w[f"{p}.ffn.down"], w[f"{p}.ffn.down_bias"],
            jnp.int32(pos),
            all_k_caches[layer], all_v_caches[layer],
            kernel=_decode_layer_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((d_model,), jnp.float32),
                jax.ShapeDtypeStruct((n_heads, d_head), jnp.bfloat16),
                jax.ShapeDtypeStruct((n_heads, d_head), jnp.bfloat16),
            ],
            grid=(1,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
            N_HEADS=n_heads, MAX_SEQ=max_seq,
        )
        h = h_out
        new_k_caches.append(all_k_caches[layer].at[:, pos, :].set(k_new))
        new_v_caches.append(all_v_caches[layer].at[:, pos, :].set(v_new))

    # Output projection
    (logits_pad,) = jt.triton_call(
        h,
        w["ln_final.scale"], w["ln_final.bias"],
        w["_output_proj_padded"],
        kernel=_decode_output_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
        ],
        grid=(1,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
    )

    return logits_pad[:vocab_size], new_k_caches, new_v_caches
