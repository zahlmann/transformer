"""Small decoder-only transformer in JAX (pure functional).

Phase C architecture: RMSNorm, RoPE, SwiGLU, no biases, tied embeddings.
"""

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
                     n_kv_heads=None):
    """Initialize a decoder-only transformer. Returns a flat dict of params.

    Architecture: RMSNorm, RoPE, SwiGLU FFN, no biases, tied embeddings.
    n_kv_heads: number of KV heads for GQA. If None, defaults to n_heads (standard MHA).
    """
    assert d_model % n_heads == 0
    d_head = d_model // n_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads
    assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

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
    }

    # token embedding: (vocab_size, d_model) — also used as output projection (tied)
    key, k = jax.random.split(key)
    params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02

    for layer in range(n_layers):
        prefix = f"layer{layer}"

        # RMSNorm 1 (pre-attention) — scale only, no bias
        params[f"{prefix}.ln1.scale"] = jnp.ones(d_model)

        # attention: Q, O use n_heads; K, V use n_kv_heads (GQA). No biases.
        d_kv = n_kv_heads * d_head
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.q"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.k"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.v"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

        # RMSNorm 2 (pre-FFN) — scale only, no bias
        params[f"{prefix}.ln2.scale"] = jnp.ones(d_model)

        # SwiGLU FFN: gate (d -> d_ff), up (d -> d_ff), down (d_ff -> d). No biases.
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.gate"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.up"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.down"] = jax.random.normal(k, (d_ff, d_model)) * (d_ff ** -0.5)

    # final RMSNorm — scale only, no bias
    params["ln_final.scale"] = jnp.ones(d_model)

    # no separate output_proj — tied to token_emb

    return params, config


def rms_norm(x, scale, eps=1e-5):
    """RMSNorm: scale * x / sqrt(mean(x²) + eps)."""
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return scale * x / rms


def precompute_rope_table(context_len, d_head, base=10000.0):
    """Precompute cos/sin tables for RoPE. Returns (context_len, d_head//2) each."""
    half = d_head // 2
    freqs = base ** (-jnp.arange(0, half, dtype=jnp.float32) * 2.0 / d_head)  # (half,)
    positions = jnp.arange(context_len, dtype=jnp.float32)  # (context_len,)
    angles = positions[:, None] * freqs[None, :]  # (context_len, half)
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE to x. x: (..., d_head), cos/sin: (seq, d_head//2) or (d_head//2,)."""
    half = x.shape[-1] // 2
    x_even = x[..., :half]
    x_odd = x[..., half:]
    return jnp.concatenate([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ], axis=-1)


def causal_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin):
    """Multi-head causal self-attention with GQA and RoPE. x: (seq_len, d_model)."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    q = (x @ wq).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)     # (n_heads, seq, d_head)
    k = (x @ wk).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)  # (n_kv_heads, seq, d_head)
    v = (x @ wv).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)  # (n_kv_heads, seq, d_head)

    # apply RoPE to Q and K
    cos_seq = cos[:seq_len]  # (seq, d_head//2)
    sin_seq = sin[:seq_len]
    q = apply_rope(q, cos_seq[None, :, :], sin_seq[None, :, :])
    k = apply_rope(k, cos_seq[None, :, :], sin_seq[None, :, :])

    # GQA: repeat KV heads to match Q heads
    if n_kv_heads < n_heads:
        repeats = n_heads // n_kv_heads
        k = jnp.repeat(k, repeats, axis=0)
        v = jnp.repeat(v, repeats, axis=0)

    # scaled dot-product attention with causal mask
    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 2, 1)) * scale
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)

    out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ wo


def _transformer_layer(h, ln1_s, wq, wk, wv, wo,
                       ln2_s, ffn_gate, ffn_up, ffn_down,
                       n_heads, n_kv_heads, context_len, d_head):
    """Single transformer layer: RMSNorm -> Attn -> RMSNorm -> SwiGLU FFN.

    RoPE tables are recomputed inside the layer (cheap, avoids saving through checkpoint).
    """
    cos, sin = precompute_rope_table(context_len, d_head)
    h_norm = rms_norm(h, ln1_s)
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin)
    h = h + attn_out

    h_norm2 = rms_norm(h, ln2_s)
    # SwiGLU: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
    h = h + h_ff
    return h


def transformer_forward(params, config, x):
    """Forward pass. x: (seq_len,) integer token indices. Returns logits (seq_len, vocab_size)."""
    h = params["token_emb"][x]  # no positional embedding — RoPE is applied in attention
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_head = config["d_head"]
    context_len = config["context_len"]

    for layer in range(config["n_layers"]):
        p = f"layer{layer}"
        h = jax.checkpoint(_transformer_layer, static_argnums=(10, 11, 12, 13))(
            h,
            params[f"{p}.ln1.scale"],
            params[f"{p}.attn.q"], params[f"{p}.attn.k"],
            params[f"{p}.attn.v"], params[f"{p}.attn.o"],
            params[f"{p}.ln2.scale"],
            params[f"{p}.ffn.gate"], params[f"{p}.ffn.up"], params[f"{p}.ffn.down"],
            n_heads, n_kv_heads, context_len, d_head,
        )

    h = rms_norm(h, params["ln_final.scale"])
    logits = h @ params["token_emb"].T  # tied embeddings
    return logits


def prefill_with_kv(params, config, x):
    """Forward pass that also returns KV caches for decode. Works with MHA and GQA."""
    seq_len = x.shape[0]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_head = config["d_head"]
    max_seq = config["context_len"]

    cos, sin = precompute_rope_table(max_seq, d_head)

    h = params["token_emb"][x]  # no positional embedding
    k_caches, v_caches = [], []

    for layer in range(config["n_layers"]):
        prefix = f"layer{layer}"
        h_norm = rms_norm(h, params[f"{prefix}.ln1.scale"])

        # project K/V, apply RoPE to K, store in cache
        k_proj = (h_norm @ params[f"{prefix}.attn.k"]).reshape(seq_len, n_kv_heads, d_head)
        v_proj = (h_norm @ params[f"{prefix}.attn.v"]).reshape(seq_len, n_kv_heads, d_head)
        # apply RoPE to K
        cos_seq = cos[:seq_len]
        sin_seq = sin[:seq_len]
        k_proj = apply_rope(k_proj, cos_seq[:, None, :], sin_seq[:, None, :])

        k_proj_t = k_proj.transpose(1, 0, 2)  # (n_kv_heads, seq, d_head)
        v_proj_t = v_proj.transpose(1, 0, 2)

        k_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
        k_cache = k_cache.at[:, :seq_len, :].set(k_proj_t.astype(jnp.bfloat16))
        v_cache = v_cache.at[:, :seq_len, :].set(v_proj_t.astype(jnp.bfloat16))
        k_caches.append(k_cache)
        v_caches.append(v_cache)

        # full attention with RoPE
        attn_out = causal_attention(
            h_norm, params[f"{prefix}.attn.q"], params[f"{prefix}.attn.k"],
            params[f"{prefix}.attn.v"], params[f"{prefix}.attn.o"],
            n_heads, n_kv_heads, cos, sin)

        h = h + attn_out
        h_norm2 = rms_norm(h, params[f"{prefix}.ln2.scale"])
        # SwiGLU FFN
        h_ff = (jax.nn.silu(h_norm2 @ params[f"{prefix}.ffn.gate"]) *
                (h_norm2 @ params[f"{prefix}.ffn.up"])) @ params[f"{prefix}.ffn.down"]
        h = h + h_ff

    h = rms_norm(h, params["ln_final.scale"])
    logits = h @ params["token_emb"].T  # tied embeddings
    return logits, k_caches, v_caches


def transformer_forward_batch(params, config, x_batch):
    """Batched forward pass. x_batch: (batch, seq_len). Returns (batch, seq_len, vocab)."""
    return jax.vmap(lambda x: transformer_forward(params, config, x))(x_batch)


def cross_entropy_loss(logits, targets):
    """Mean cross-entropy loss. logits: (batch, seq, vocab), targets: (batch, seq)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)


def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))
