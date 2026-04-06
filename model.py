"""Small decoder-only transformer in JAX (pure functional).

Architecture: RMSNorm, RoPE, GQA, SwiGLU, no biases, tied embeddings.
Supports multi-token prediction (MTP) heads.
"""

import functools

import jax
import jax.numpy as jnp


def _swiglu_d_ff(d_model):
    """Compute SwiGLU FFN hidden dim: match param count of standard 4d FFN.

    Standard FFN: 2 * d * 4d = 8d² params
    SwiGLU FFN:   3 * d * d_ff params
    Match: d_ff = 8d/3, rounded up to multiple of 128.
    """
    return ((8 * d_model // 3 + 127) // 128) * 128


def init_transformer(key, vocab_size, d_model=64, n_heads=2, n_layers=1, context_len=128,
                     n_kv_heads=2, n_mtp_heads=0):
    """Initialize a decoder-only transformer. Returns (params dict, config dict).

    n_kv_heads: number of KV heads for GQA.
    n_mtp_heads: number of extra multi-token prediction heads.
    """
    assert d_model % n_heads == 0
    d_head = d_model // n_heads
    assert n_heads % n_kv_heads == 0

    d_ff = _swiglu_d_ff(d_model)

    params = {}
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "d_head": d_head,
        "d_ff": d_ff,
        "context_len": context_len,
        "n_mtp_heads": n_mtp_heads,
    }

    # token embedding: (vocab_size, d_model) — also used as output projection (tied)
    key, k = jax.random.split(key)
    params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02

    d_kv = n_kv_heads * d_head

    for layer in range(n_layers):
        prefix = f"layer{layer}"

        # RMSNorm 1 (pre-attention)
        params[f"{prefix}.ln1.scale"] = jnp.ones(d_model)

        # Q, O use n_heads; K, V use n_kv_heads (GQA)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.q"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.k"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.v"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

        # RMSNorm 2 (pre-FFN)
        params[f"{prefix}.ln2.scale"] = jnp.ones(d_model)

        # SwiGLU FFN
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.gate"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.up"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.down"] = jax.random.normal(k, (d_ff, d_model)) * (d_ff ** -0.5)

    # final RMSNorm
    params["ln_final.scale"] = jnp.ones(d_model)

    # MTP heads: small projection (d_model -> d_model) per extra head
    for k_idx in range(n_mtp_heads):
        key, k1, k2 = jax.random.split(key, 3)
        params[f"mtp.{k_idx}.proj"] = jax.random.normal(k1, (d_model, d_model)) * (d_model ** -0.5)
        params[f"mtp.{k_idx}.norm"] = jnp.ones(d_model)

    return params, config


def rms_norm(x, scale, eps=1e-5):
    """RMSNorm: scale * x / sqrt(mean(x^2) + eps)."""
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps).astype(x.dtype)
    return scale * (x / rms)


def precompute_rope_table(context_len, d_head, base=10000.0):
    """Precompute cos/sin tables for RoPE. Returns (context_len, d_head//2) each."""
    half = d_head // 2
    freqs = base ** (-jnp.arange(0, half, dtype=jnp.float32) * 2.0 / d_head)
    positions = jnp.arange(context_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE to x. x: (..., d_head), cos/sin: (seq, d_head//2) or (d_head//2,)."""
    half = x.shape[-1] // 2
    x_even, x_odd = x[..., :half], x[..., half:]
    return jnp.concatenate([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ], axis=-1)


# ─── causal attention ───

def causal_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin):
    """Multi-head causal self-attention with GQA and RoPE. x: (seq_len, d_model)."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    q = (x @ wq).reshape(seq_len, n_heads, d_head)
    k = (x @ wk).reshape(seq_len, n_kv_heads, d_head)
    v = (x @ wv).reshape(seq_len, n_kv_heads, d_head)

    cos_seq = cos[:seq_len]
    sin_seq = sin[:seq_len]
    q = apply_rope(q.transpose(1, 0, 2), cos_seq[None, :, :], sin_seq[None, :, :]).transpose(1, 0, 2)
    k = apply_rope(k.transpose(1, 0, 2), cos_seq[None, :, :], sin_seq[None, :, :]).transpose(1, 0, 2)

    # cuDNN FlashAttention when bf16, XLA fallback for f32 (inference/testing)
    if v.dtype == jnp.bfloat16:
        q = q.astype(jnp.bfloat16)
        k = k.astype(jnp.bfloat16)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation='cudnn')
    else:
        v = v.astype(q.dtype)
        out = jax.nn.dot_product_attention(q, k, v, is_causal=True)

    return out.reshape(seq_len, d_model) @ wo


# ─── transformer layers ───

def _attn_layer(h, ln1_s, wq, wk, wv, wo,
                ln2_s, ffn_gate, ffn_up, ffn_down,
                n_heads, n_kv_heads, context_len, d_head):
    """Standard attention transformer layer."""
    cos, sin = precompute_rope_table(context_len, d_head)
    h_norm = rms_norm(h, ln1_s)
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin)
    h = h + attn_out
    h_norm2 = rms_norm(h, ln2_s)
    h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
    return h + h_ff


def _transformer_trunk(params, config, x):
    """Run transformer layers. Returns final hidden states (seq_len, d_model)."""
    h = params["token_emb"][x]
    n_heads = config["n_heads"]
    n_kv_heads = config["n_kv_heads"]
    d_head = config["d_head"]
    context_len = config["context_len"]
    use_checkpoint = config.get("gradient_checkpoint", True)

    maybe_checkpoint = jax.checkpoint if use_checkpoint else lambda f, **kw: f

    for layer in range(config["n_layers"]):
        p = f"layer{layer}"
        h = maybe_checkpoint(_attn_layer, static_argnums=(10, 11, 12, 13))(
            h,
            params[f"{p}.ln1.scale"],
            params[f"{p}.attn.q"], params[f"{p}.attn.k"],
            params[f"{p}.attn.v"], params[f"{p}.attn.o"],
            params[f"{p}.ln2.scale"],
            params[f"{p}.ffn.gate"], params[f"{p}.ffn.up"], params[f"{p}.ffn.down"],
            n_heads, n_kv_heads, context_len, d_head,
        )

    return rms_norm(h, params["ln_final.scale"])


def transformer_forward(params, config, x):
    """Forward pass. x: (seq_len,) integer token indices. Returns logits (seq_len, vocab_size)."""
    h = _transformer_trunk(params, config, x)
    return h @ params["token_emb"].T


def prefill_with_kv(params, config, x):
    """Forward pass returning KV caches for decode. Used by inference kernels."""
    seq_len = x.shape[0]
    n_heads, n_kv_heads = config["n_heads"], config["n_kv_heads"]
    d_head, max_seq = config["d_head"], config["context_len"]
    cos, sin = precompute_rope_table(max_seq, d_head)

    h = params["token_emb"][x]
    k_caches, v_caches = [], []

    for layer in range(config["n_layers"]):
        p = f"layer{layer}"
        h_norm = rms_norm(h, params[f"{p}.ln1.scale"])

        k_proj = (h_norm @ params[f"{p}.attn.k"]).reshape(seq_len, n_kv_heads, d_head)
        v_proj = (h_norm @ params[f"{p}.attn.v"]).reshape(seq_len, n_kv_heads, d_head)
        k_proj = apply_rope(k_proj, cos[:seq_len, None, :], sin[:seq_len, None, :])

        k_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
        k_cache = k_cache.at[:, :seq_len, :].set(k_proj.transpose(1, 0, 2).astype(jnp.bfloat16))
        v_cache = v_cache.at[:, :seq_len, :].set(v_proj.transpose(1, 0, 2).astype(jnp.bfloat16))
        k_caches.append(k_cache)
        v_caches.append(v_cache)

        attn_out = causal_attention(
            h_norm, params[f"{p}.attn.q"], params[f"{p}.attn.k"],
            params[f"{p}.attn.v"], params[f"{p}.attn.o"],
            n_heads, n_kv_heads, cos, sin)
        h = h + attn_out

        h_norm2 = rms_norm(h, params[f"{p}.ln2.scale"])
        h_ff = (jax.nn.silu(h_norm2 @ params[f"{p}.ffn.gate"]) *
                (h_norm2 @ params[f"{p}.ffn.up"])) @ params[f"{p}.ffn.down"]
        h = h + h_ff

    logits = rms_norm(h, params["ln_final.scale"]) @ params["token_emb"].T
    return logits, k_caches, v_caches


def transformer_forward_batch(params, config, x_batch):
    """Batched forward pass. x_batch: (batch, seq_len). Returns (batch, seq_len, vocab)."""
    return jax.vmap(lambda x: transformer_forward(params, config, x))(x_batch)


def transformer_loss_fused(params, config, x_batch, targets, chunk_size=4096):
    """Fused forward + cross-entropy loss without materializing full logits.

    x_batch: (batch, seq_len), targets: (batch, seq_len).
    Tiles the output projection + softmax + CE over vocab chunks.
    Critical for vocab >= 16K (avoids OOM from full logits tensor).
    Supports MTP: computes loss for next-token + extra prediction heads.
    """
    h_batch = jax.vmap(lambda x: _transformer_trunk(params, config, x))(x_batch)
    token_emb = params["token_emb"]

    # standard next-token loss (head 0)
    loss = fused_output_and_loss(h_batch, token_emb, targets, chunk_size)

    # MTP extra heads: predict tokens t+k using h @ mtp_proj_k
    n_mtp = config.get("n_mtp_heads", 0)
    for k_idx in range(n_mtp):
        shift = k_idx + 1
        h_shifted = h_batch[:, :-shift, :]
        tgt_shifted = targets[:, shift:]

        proj = params[f"mtp.{k_idx}.proj"]
        norm_scale = params[f"mtp.{k_idx}.norm"]
        h_proj = rms_norm(h_shifted @ proj, norm_scale)

        mtp_loss = fused_output_and_loss(h_proj, token_emb, tgt_shifted, chunk_size)
        loss = loss + mtp_loss

    return loss / (1 + n_mtp)


def cross_entropy_loss(logits, targets):
    """Mean cross-entropy loss. logits: (batch, seq, vocab), targets: (batch, seq)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)


# ─── fused cross-entropy (memory-efficient for large vocab) ───

def _chunked_ce_fwd(h, weight, targets, chunk_size):
    """Forward: compute CE loss without materializing full (N, vocab) logits.

    Tiles over vocab dimension, accumulates logsumexp online.
    h: (N, d_model), weight: (vocab, d_model), targets: (N,).
    """
    N, d = h.shape
    vocab = weight.shape[0]

    target_emb = weight[targets]
    target_logits = jnp.sum(h * target_emb, axis=-1)

    max_logit = jnp.full(N, -1e30)
    sum_exp = jnp.zeros(N)

    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        chunk_logits = h @ weight[start:end].T
        chunk_max = chunk_logits.max(axis=-1)
        new_max = jnp.maximum(max_logit, chunk_max)
        sum_exp = sum_exp * jnp.exp(max_logit - new_max) + jnp.sum(jnp.exp(chunk_logits - new_max[:, None]), axis=-1)
        max_logit = new_max

    logsumexp = max_logit + jnp.log(sum_exp)
    return jnp.mean(logsumexp - target_logits)


@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def fused_cross_entropy(h, weight, targets, chunk_size):
    """Fused output projection + cross-entropy without materializing logits.

    h: (N, d_model), weight: (vocab, d_model), targets: (N,), chunk_size: int.
    Returns scalar loss.
    """
    return _chunked_ce_fwd(h, weight, targets, chunk_size)


def _fused_ce_fwd(h, weight, targets, chunk_size):
    loss = _chunked_ce_fwd(h, weight, targets, chunk_size)
    return loss, (h, weight, targets)


def _fused_ce_bwd(chunk_size, res, g):
    """Backward: compute grads w.r.t. h and weight without materializing full logits.

    Computation in f32 for numerical stability, results cast to input dtypes.
    """
    h, weight, targets = res
    h_dtype, w_dtype = h.dtype, weight.dtype
    h = h.astype(jnp.float32)
    weight = weight.astype(jnp.float32)
    N, d = h.shape
    vocab = weight.shape[0]
    scale = g / N

    # recompute logsumexp (needed for softmax)
    max_logit = jnp.full(N, -1e30)
    sum_exp = jnp.zeros(N)
    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        chunk_logits = h @ weight[start:end].T
        chunk_max = chunk_logits.max(axis=-1)
        new_max = jnp.maximum(max_logit, chunk_max)
        sum_exp = sum_exp * jnp.exp(max_logit - new_max) + jnp.sum(jnp.exp(chunk_logits - new_max[:, None]), axis=-1)
        max_logit = new_max
    logsumexp = max_logit + jnp.log(sum_exp)

    # accumulate grad_h and grad_weight in chunks
    grad_h = jnp.zeros_like(h)
    grad_weight = jnp.zeros_like(weight)

    for start in range(0, vocab, chunk_size):
        end = min(start + chunk_size, vocab)
        w_chunk = weight[start:end]
        chunk_logits = h @ w_chunk.T
        softmax_chunk = jnp.exp(chunk_logits - logsumexp[:, None])

        grad_h = grad_h + (softmax_chunk @ w_chunk) * scale
        grad_weight = grad_weight.at[start:end].add((softmax_chunk.T @ h) * scale)

    # subtract target contribution
    target_emb = weight[targets]
    grad_h = grad_h - target_emb * scale
    grad_weight = grad_weight.at[targets].add(-h * scale)

    return grad_h.astype(h_dtype), grad_weight.astype(w_dtype), None


fused_cross_entropy.defvjp(_fused_ce_fwd, _fused_ce_bwd)


def fused_output_and_loss(h, token_emb, targets, chunk_size=4096):
    """Fused final output projection + CE loss for large vocabs.

    h: (batch, seq, d_model), token_emb: (vocab, d_model), targets: (batch, seq).
    Returns scalar mean loss.
    """
    batch, seq, d = h.shape
    h_flat = h.reshape(batch * seq, d)
    targets_flat = targets.reshape(batch * seq)
    return fused_cross_entropy(h_flat, token_emb, targets_flat, chunk_size)


def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))
