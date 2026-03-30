"""
Fused transformer prefill kernel — entire forward pass in one kernel call.

Processes full 128-token sequence: embedding → LN → attention → FFN → LN → logits.
Outputs logits + KV cache for decode phase. All weights stay in registers.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

SEQ        = tl.constexpr(128)
D_MODEL    = tl.constexpr(64)
D_HEAD     = tl.constexpr(32)
D_FF       = tl.constexpr(256)
BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)  # tile size for output projection loop


@triton.jit
def _prefill_kernel(
    # Weights (bf16)
    token_emb_ptr, pos_emb_ptr,
    ln1_scale_ptr, ln1_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,
    ln2_scale_ptr, ln2_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr, ffn_down_ptr, ffn_down_bias_ptr,
    ln_final_scale_ptr, ln_final_bias_ptr,
    output_proj_ptr,
    # Input
    x_ptr,
    # Outputs
    logits_ptr, k_cache_ptr, v_cache_ptr,
    # Constexpr parameters for variable vocab
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    pos = tl.arange(0, SEQ)
    d = tl.arange(0, D_MODEL)

    # ── Embedding ──
    tokens = tl.load(x_ptr + pos)
    h = (tl.load(token_emb_ptr + tokens[:, None] * D_MODEL + d[None, :]).to(tl.float32)
       + tl.load(pos_emb_ptr + pos[:, None] * D_MODEL + d[None, :]).to(tl.float32))

    # ── Layer Norm 1 ──
    ln1_s = tl.load(ln1_scale_ptr + d).to(tl.float32)
    ln1_b = tl.load(ln1_bias_ptr + d).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = h - mean
    h_norm = ln1_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln1_b[None, :]

    # ── Multi-head Attention ──
    scale = 0.17677669529663689  # 1/sqrt(32)
    dh = tl.arange(0, D_HEAD)

    for head in tl.range(2):
        hd = head * D_HEAD + dh

        K = tl.dot(h_norm.to(tl.bfloat16), tl.load(wk_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)).to(tl.float32)
        V = tl.dot(h_norm.to(tl.bfloat16), tl.load(wv_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)).to(tl.float32)
        Q = tl.dot(h_norm.to(tl.bfloat16), tl.load(wq_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)).to(tl.float32)

        # Save KV cache
        tl.store(k_cache_ptr + head * SEQ * D_HEAD + pos[:, None] * D_HEAD + dh[None, :], K.to(tl.bfloat16))
        tl.store(v_cache_ptr + head * SEQ * D_HEAD + pos[:, None] * D_HEAD + dh[None, :], V.to(tl.bfloat16))

        # Causal attention
        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        scores = tl.where(pos[:, None] >= pos[None, :], scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        attn = exp_s / tl.sum(exp_s, axis=1)[:, None]

        # Attention output → O projection → residual
        attn_out = tl.dot(attn.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)
        h += tl.dot(attn_out.to(tl.bfloat16), tl.load(wo_ptr + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)).to(tl.float32)

    # ── Layer Norm 2 ──
    ln2_s = tl.load(ln2_scale_ptr + d).to(tl.float32)
    ln2_b = tl.load(ln2_bias_ptr + d).to(tl.float32)
    mean2 = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc2 = h - mean2
    h_norm2 = ln2_s[None, :] * hc2 * tl.math.rsqrt(tl.sum(hc2 * hc2, axis=1)[:, None] / D_MODEL + 1e-5) + ln2_b[None, :]

    # ── FFN (tiled to avoid register blowup) ──
    ffn_out = tl.zeros((SEQ, D_MODEL), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        up = tl.dot(h_norm2.to(tl.bfloat16), tl.load(ffn_up_ptr + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)).to(tl.float32)
        up += tl.load(ffn_up_bias_ptr + kk).to(tl.float32)[None, :]
        act = up * tl.sigmoid(1.702 * up)  # GELU approximation
        ffn_out += tl.dot(act.to(tl.bfloat16), tl.load(ffn_down_ptr + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)).to(tl.float32)
    h = h + ffn_out + tl.load(ffn_down_bias_ptr + d).to(tl.float32)[None, :]

    # ── Final Layer Norm ──
    lnf_s = tl.load(ln_final_scale_ptr + d).to(tl.float32)
    lnf_b = tl.load(ln_final_bias_ptr + d).to(tl.float32)
    mean_f = tl.sum(h, axis=1)[:, None] / D_MODEL
    hcf = h - mean_f
    h_final = lnf_s[None, :] * hcf * tl.math.rsqrt(tl.sum(hcf * hcf, axis=1)[:, None] / D_MODEL + 1e-5) + lnf_b[None, :]

    # ── Output Projection (tiled over vocab dimension) ──
    for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
        vv = v_start + tl.arange(0, VOCAB_TILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final.to(tl.bfloat16), out_w).to(tl.float32)
        tile_logits = tl.where(vv[None, :] < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + pos[:, None] * VOCAB_PAD + vv[None, :], tile_logits)


def fused_prefill(params, x, vocab_size=65):
    """Run full prefill in one fused kernel call.

    Args:
        params: weight dict from model.init_transformer
        x: (128,) int32 token IDs
        vocab_size: actual vocabulary size

    Returns:
        logits:  (128, vocab_size) float32
        k_cache: (2, 128, 32) bf16
        v_cache: (2, 128, 32) bf16
    """
    assert x.shape == (128,)
    vocab_pad = ((vocab_size + 127) // 128) * 128  # round up to VOCAB_TILE

    def bf(key):
        return params[key].astype(jnp.bfloat16)

    pad_v = vocab_pad - params["output_proj"].shape[1]
    output_proj_padded = jnp.pad(params["output_proj"], [(0, 0), (0, pad_v)]).astype(jnp.bfloat16)

    logits_pad, k_cache, v_cache = jt.triton_call(
        bf("token_emb"), bf("pos_emb"),
        bf("layer0.ln1.scale"), bf("layer0.ln1.bias"),
        bf("layer0.attn.q"), bf("layer0.attn.k"),
        bf("layer0.attn.v"), bf("layer0.attn.o"),
        bf("layer0.ln2.scale"), bf("layer0.ln2.bias"),
        bf("layer0.ffn.up"), bf("layer0.ffn.up_bias"),
        bf("layer0.ffn.down"), bf("layer0.ffn.down_bias"),
        bf("ln_final.scale"), bf("ln_final.bias"),
        output_proj_padded,
        x.astype(jnp.int32),
        kernel=_prefill_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((128, vocab_pad), jnp.float32),
            jax.ShapeDtypeStruct((2, 128, 32), jnp.bfloat16),
            jax.ShapeDtypeStruct((2, 128, 32), jnp.bfloat16),
        ],
        grid=(1,),
        num_warps=4,
        num_stages=1,
        VOCAB_SIZE=vocab_size,
        VOCAB_PAD=vocab_pad,
    )
    return logits_pad[:, :vocab_size], k_cache, v_cache
