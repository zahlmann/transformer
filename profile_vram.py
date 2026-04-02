"""Profile VRAM usage for different model configurations.

Tests several (d_model, n_layers) configs with forward + backward pass
to find the largest model that fits in ~12-14 GB VRAM for training.

Usage: uv run profile_vram.py
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), ".jax_cache")

import jax
import jax.numpy as jnp
import optax
import time

from model import init_transformer, transformer_forward_batch, cross_entropy_loss, count_params


def profile_config(d_model, n_heads, n_kv_heads, n_layers, context_len, batch_size, vocab_size):
    """Profile a single model configuration. Returns (params_count, peak_vram_mb) or None if OOM."""
    key = jax.random.key(42)
    try:
        params, config = init_transformer(
            key, vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, context_len=context_len, n_kv_heads=n_kv_heads)

        n_params = count_params(params)

        # Set up optimizer (same as train.py)
        optimizer = optax.adamw(1e-4, weight_decay=0.1)
        opt_state = optimizer.init(params)

        # Create dummy batch
        x = jax.random.randint(jax.random.key(0), (batch_size, context_len), 0, vocab_size)
        y = jax.random.randint(jax.random.key(1), (batch_size, context_len), 0, vocab_size)

        @jax.jit
        def train_step(params, opt_state, x, y):
            def loss_fn(params):
                params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
                logits = transformer_forward_batch(params_bf16, config, x)
                return cross_entropy_loss(logits.astype(jnp.float32), y)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), new_opt_state, loss

        # Warm up (triggers JIT compilation)
        params, opt_state, loss = train_step(params, opt_state, x, y)
        jax.block_until_ready(loss)

        # Run a few steps to get stable memory reading
        for _ in range(3):
            params, opt_state, loss = train_step(params, opt_state, x, y)
        jax.block_until_ready(loss)

        mem = jax.local_devices()[0].memory_stats()
        peak_mb = mem["peak_bytes_in_use"] / 1e6

        # Clean up
        del params, opt_state, x, y
        jax.clear_caches()

        return n_params, peak_mb

    except Exception as e:
        if "out of memory" in str(e).lower() or "resource" in str(e).lower():
            jax.clear_caches()
            return None, None
        raise


def main():
    vocab_size = 32000
    context_len = 512

    configs = [
        # (d_model, n_heads, n_kv_heads, n_layers, batch_size)
        # Current model
        (768, 24, 6, 12, 16),
        (768, 24, 6, 12, 32),
        # Scale up candidates
        (1024, 16, 4, 16, 16),
        (1024, 16, 4, 16, 32),
        (1024, 16, 4, 20, 16),
        (1024, 16, 4, 24, 16),
        (1024, 32, 8, 16, 16),
        (1280, 20, 4, 16, 16),
        (1280, 20, 4, 20, 16),
        (1536, 24, 6, 16, 16),
        (1536, 24, 6, 16, 8),
    ]

    results = []
    print(f"{'Config':40s} {'Params':>10s} {'VRAM':>10s} {'Util':>6s}")
    print("-" * 70)

    for d, h, kv_h, l, bs in configs:
        label = f"d={d} h={h} kv={kv_h} l={l} bs={bs}"
        print(f"  Testing {label}...", end="", flush=True)

        n_params, peak_mb = profile_config(d, h, kv_h, l, context_len, bs, vocab_size)

        if n_params is None:
            print(" OOM")
            results.append((label, "OOM", "OOM", "OOM"))
        else:
            util = peak_mb / 16000 * 100  # % of 16GB
            print(f" {n_params/1e6:.1f}M params, {peak_mb:.0f} MB ({util:.0f}%)")
            results.append((label, f"{n_params/1e6:.1f}M", f"{peak_mb:.0f}MB", f"{util:.0f}%"))

    print("\n" + "=" * 70)
    print(f"{'Config':40s} {'Params':>10s} {'VRAM':>10s} {'Util':>6s}")
    print("-" * 70)
    for label, params, vram, util in results:
        print(f"{label:40s} {params:>10s} {vram:>10s} {util:>6s}")

    # Save results
    with open("vram_profile.txt", "w") as f:
        f.write(f"{'Config':40s} {'Params':>10s} {'VRAM':>10s} {'Util':>6s}\n")
        for label, params, vram, util in results:
            f.write(f"{label:40s} {params:>10s} {vram:>10s} {util:>6s}\n")
    print("\nResults saved to vram_profile.txt")


if __name__ == "__main__":
    main()
