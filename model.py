"""Small decoder-only transformer in JAX (pure functional)."""

import jax
import jax.numpy as jnp


def init_transformer(key, vocab_size, d_model=64, n_heads=2, n_layers=1, context_len=128,
                     n_kv_heads=None, parallel_residual=False):
    """Initialize a small decoder-only transformer. Returns a flat dict of params.

    n_kv_heads: number of KV heads for GQA. If None, defaults to n_heads (standard MHA).
    parallel_residual: if True, compute attention and FFN in parallel (GPT-NeoX style).
    """
    assert d_model % n_heads == 0
    d_head = d_model // n_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads
    assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"

    params = {}
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "d_head": d_head,
        "context_len": context_len,
        "parallel_residual": parallel_residual,
    }

    # token embedding: (vocab_size, d_model)
    key, k = jax.random.split(key)
    params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02

    # learned positional embedding: (context_len, d_model)
    key, k = jax.random.split(key)
    params["pos_emb"] = jax.random.normal(k, (context_len, d_model)) * 0.02

    for layer in range(n_layers):
        prefix = f"layer{layer}"

        # layer norm 1 (pre-attention)
        params[f"{prefix}.ln1.scale"] = jnp.ones(d_model)
        params[f"{prefix}.ln1.bias"] = jnp.zeros(d_model)

        # attention: Q, O use n_heads; K, V use n_kv_heads (GQA)
        d_kv = n_kv_heads * d_head
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.q"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.k"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.v"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

        # layer norm 2 (pre-FFN)
        params[f"{prefix}.ln2.scale"] = jnp.ones(d_model)
        params[f"{prefix}.ln2.bias"] = jnp.zeros(d_model)

        # FFN: up projection (d_model -> 4*d_model), down projection (4*d_model -> d_model)
        d_ff = 4 * d_model
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.up"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)

        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.down"] = jax.random.normal(k, (d_ff, d_model)) * (d_ff ** -0.5)

    # final layer norm
    params["ln_final.scale"] = jnp.ones(d_model)
    params["ln_final.bias"] = jnp.zeros(d_model)

    # separate output projection (weight tying hurts quality with small models)
    key, k = jax.random.split(key)
    params["output_proj"] = jax.random.normal(k, (d_model, vocab_size)) * (d_model ** -0.5)

    return params, config


def layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def causal_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads=None):
    """Multi-head causal self-attention with GQA. x: (seq_len, d_model)."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads

    q = (x @ wq).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)     # (n_heads, seq, d_head)
    k = (x @ wk).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)  # (n_kv_heads, seq, d_head)
    v = (x @ wv).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)  # (n_kv_heads, seq, d_head)

    # GQA: repeat KV heads to match Q heads
    if n_kv_heads < n_heads:
        repeats = n_heads // n_kv_heads
        k = jnp.repeat(k, repeats, axis=0)  # (n_heads, seq, d_head)
        v = jnp.repeat(v, repeats, axis=0)

    # scaled dot-product attention with causal mask
    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 2, 1)) * scale  # (n_heads, seq, seq)
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)

    out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, d_model)  # (seq, d_model)
    return out @ wo


def _transformer_layer(h, ln1_s, ln1_b, wq, wk, wv, wo,
                       ln2_s, ln2_b, ffn_up, ffn_down,
                       n_heads, n_kv_heads, parallel):
    """Single transformer layer with explicit params (for gradient checkpointing)."""
    h_norm = layer_norm(h, ln1_s, ln1_b)
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads, n_kv_heads)

    if parallel:
        h_norm2 = layer_norm(h, ln2_s, ln2_b)
        h_ff = jax.nn.gelu(h_norm2 @ ffn_up)
        h_ff = h_ff @ ffn_down
        h = h + attn_out + h_ff
    else:
        h = h + attn_out
        h_norm2 = layer_norm(h, ln2_s, ln2_b)
        h_ff = jax.nn.gelu(h_norm2 @ ffn_up)
        h_ff = h_ff @ ffn_down
        h = h + h_ff
    return h


def transformer_forward(params, config, x):
    """Forward pass. x: (seq_len,) integer token indices. Returns logits (seq_len, vocab_size)."""
    seq_len = x.shape[0]
    h = params["token_emb"][x] + params["pos_emb"][:seq_len]
    parallel = config.get("parallel_residual", False)
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)

    for layer in range(config["n_layers"]):
        p = f"layer{layer}"
        # Gradient checkpointing: pass only this layer's params to avoid saving
        # the full model params dict at each checkpoint boundary.
        h = jax.checkpoint(_transformer_layer, static_argnums=(11, 12, 13))(
            h,
            params[f"{p}.ln1.scale"], params[f"{p}.ln1.bias"],
            params[f"{p}.attn.q"], params[f"{p}.attn.k"],
            params[f"{p}.attn.v"], params[f"{p}.attn.o"],
            params[f"{p}.ln2.scale"], params[f"{p}.ln2.bias"],
            params[f"{p}.ffn.up"], params[f"{p}.ffn.down"],
            n_heads, n_kv_heads, parallel,
        )

    h = layer_norm(h, params["ln_final.scale"], params["ln_final.bias"])
    logits = h @ params["output_proj"]
    return logits


def prefill_with_kv(params, config, x):
    """Forward pass that also returns KV caches for decode. Works with MHA and GQA."""
    seq_len = x.shape[0]
    n_kv_heads = config.get("n_kv_heads", config["n_heads"])
    d_head = config["d_head"]
    max_seq = config["context_len"]

    h = params["token_emb"][x] + params["pos_emb"][:seq_len]
    k_caches, v_caches = [], []

    for layer in range(config["n_layers"]):
        prefix = f"layer{layer}"
        h_norm = layer_norm(h, params[f"{prefix}.ln1.scale"], params[f"{prefix}.ln1.bias"])

        # Extract KV cache before attention
        k_proj = (h_norm @ params[f"{prefix}.attn.k"]).reshape(seq_len, n_kv_heads, d_head)
        v_proj = (h_norm @ params[f"{prefix}.attn.v"]).reshape(seq_len, n_kv_heads, d_head)
        k_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
        v_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
        k_cache = k_cache.at[:, :seq_len, :].set(k_proj.transpose(1, 0, 2).astype(jnp.bfloat16))
        v_cache = v_cache.at[:, :seq_len, :].set(v_proj.transpose(1, 0, 2).astype(jnp.bfloat16))
        k_caches.append(k_cache)
        v_caches.append(v_cache)

        # Attention
        attn_out = causal_attention(
            h_norm, params[f"{prefix}.attn.q"], params[f"{prefix}.attn.k"],
            params[f"{prefix}.attn.v"], params[f"{prefix}.attn.o"],
            config["n_heads"], n_kv_heads)

        # FFN (parallel or sequential residual)
        parallel = config.get("parallel_residual", False)
        if parallel:
            h_norm2 = layer_norm(h, params[f"{prefix}.ln2.scale"], params[f"{prefix}.ln2.bias"])
            h_ff = jax.nn.gelu(h_norm2 @ params[f"{prefix}.ffn.up"])
            h_ff = h_ff @ params[f"{prefix}.ffn.down"]
            h = h + attn_out + h_ff
        else:
            h = h + attn_out
            h_norm2 = layer_norm(h, params[f"{prefix}.ln2.scale"], params[f"{prefix}.ln2.bias"])
            h_ff = jax.nn.gelu(h_norm2 @ params[f"{prefix}.ffn.up"])
            h_ff = h_ff @ params[f"{prefix}.ffn.down"]
            h = h + h_ff

    h = layer_norm(h, params["ln_final.scale"], params["ln_final.bias"])
    logits = h @ params["output_proj"]
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
