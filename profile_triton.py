"""Profile the Triton-based EGGROLL training to find bottlenecks."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import numpy as np

from data import prepare_data
from model import init_transformer
from train_eggroll_optimized import build_param_spec
from kernels.fused_transformer_ce import fused_transformer_ce_both

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
HALF_POP = 4096
SIGMA = 0.02
ALPHA = 0.50
TEMPERATURE = 2.0

def main():
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]
    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    spec, total_vec_dim = build_param_spec(params)

    x = jnp.array(data["train_x"][:BATCH_SIZE])
    y = jnp.array(data["train_y"][:BATCH_SIZE])

    # --- Profile: random vector generation ---
    @jax.jit
    def gen_vecs(key):
        return jax.random.normal(key, (HALF_POP, total_vec_dim))

    print("Profiling random vector generation...")
    key, k = jax.random.split(key)
    vecs = gen_vecs(k)
    vecs.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(20):
        key, k = jax.random.split(key)
        vecs = gen_vecs(k)
        vecs.block_until_ready()
    t_gen = (time.perf_counter() - t0) / 20
    print(f"  Vector gen: {t_gen*1000:.1f}ms")

    # --- Profile: Triton kernel ---
    @jax.jit
    def run_kernel(params, vecs, x, y):
        return fused_transformer_ce_both(
            params["token_emb"], params["pos_emb"],
            params["layer0.ln1.scale"], params["layer0.ln1.bias"],
            params["layer0.attn.q"], params["layer0.attn.k"],
            params["layer0.attn.v"], params["layer0.attn.o"],
            params["layer0.ln2.scale"], params["layer0.ln2.bias"],
            params["layer0.ffn.up"], params["layer0.ffn.up_bias"],
            params["layer0.ffn.down"], params["layer0.ffn.down_bias"],
            params["ln_final.scale"], params["ln_final.bias"],
            params["output_proj"],
            vecs, x, y, SIGMA, ALPHA, TEMPERATURE,
        )

    print("\nProfiling Triton kernel...")
    ce_pos, ce_neg = run_kernel(params, vecs, x, y)
    ce_neg.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        ce_pos, ce_neg = run_kernel(params, vecs, x, y)
        ce_neg.block_until_ready()
    t_kernel = (time.perf_counter() - t0) / 10
    print(f"  Kernel: {t_kernel*1000:.1f}ms")

    # --- Profile: gradient computation ---
    N_SUBGROUPS = 8
    CLIP_RANGE = 2.0

    @jax.jit
    def compute_gradient(vecs, ce_pos, ce_neg):
        fp = ce_pos.sum(axis=1) / x.shape[0]
        fn = ce_neg.sum(axis=1) / x.shape[0]
        diffs = fp - fn
        # Winsorized z-score
        group_size = diffs.shape[0] // N_SUBGROUPS
        groups = diffs[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
        means = jnp.mean(groups, axis=1, keepdims=True)
        stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
        z = (groups - means) / stds
        z = jnp.clip(z, -CLIP_RANGE, CLIP_RANGE)
        shaped = z.reshape(-1)
        scale = 1.0 / (2.0 * SIGMA * HALF_POP)
        grads = {}
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v_pert = vecs[:, offset:offset + vec_dim]
            if is_2d:
                m, n = shape
                g = scale * (v_pert[:, :m] * shaped[:, None]).T @ v_pert[:, m:]
            else:
                g = scale * (v_pert * shaped[:, None]).sum(axis=0)
            grads[pkey] = g
        return grads

    print("\nProfiling gradient computation...")
    grads = compute_gradient(vecs, ce_pos, ce_neg)
    jax.block_until_ready(grads)
    t0 = time.perf_counter()
    for _ in range(20):
        grads = compute_gradient(vecs, ce_pos, ce_neg)
        jax.block_until_ready(grads)
    t_grad = (time.perf_counter() - t0) / 20
    print(f"  Gradient: {t_grad*1000:.1f}ms")

    # --- Profile: full batch (as in training) ---
    print("\nProfiling full training batch (gen + kernel + grad + adam)...")
    from train_eggroll_triton import train_one_batch, winsorized_zscore, MOMENTUM, ADAM_BETA2, ADAM_EPS, LR_START
    momentum_buf = jax.tree.map(jnp.zeros_like, params)
    v_buf = jax.tree.map(jnp.zeros_like, params)
    step = jnp.int32(0)
    lr_scale_arr = jnp.ones(len(spec))

    @jax.jit
    def train_batch(params, momentum_buf, v_buf, step, key, x, y, sigma, lr):
        return train_one_batch(params, momentum_buf, v_buf, step, key, x, y, sigma, lr)

    # warmup
    out = train_batch(params, momentum_buf, v_buf, step, key, x, y, SIGMA, LR_START)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(10):
        out = train_batch(params, momentum_buf, v_buf, step, key, x, y, SIGMA, LR_START)
        jax.block_until_ready(out)
    t_batch = (time.perf_counter() - t0) / 10
    print(f"  Full batch: {t_batch*1000:.1f}ms")
    print(f"  Breakdown: kernel={t_kernel*1000:.0f}ms + grad={t_grad*1000:.0f}ms + gen={t_gen*1000:.0f}ms = {(t_kernel+t_grad+t_gen)*1000:.0f}ms (vs {t_batch*1000:.0f}ms total)")
    print(f"  Overhead: {(t_batch - t_kernel - t_grad - t_gen)*1000:.0f}ms")

    n_batches = 61
    print(f"\n=== SUMMARY (estimated for 10 epochs) ===")
    print(f"  Kernel:   {t_kernel*n_batches*10:.1f}s ({t_kernel*n_batches*10/(t_batch*n_batches*10)*100:.0f}%)")
    print(f"  Gradient: {t_grad*n_batches*10:.1f}s ({t_grad*n_batches*10/(t_batch*n_batches*10)*100:.0f}%)")
    print(f"  Vec gen:  {t_gen*n_batches*10:.1f}s ({t_gen*n_batches*10/(t_batch*n_batches*10)*100:.0f}%)")
    print(f"  Total:    {t_batch*n_batches*10:.1f}s")

if __name__ == "__main__":
    main()
