"""Benchmark: Triton kernel vs JAX vmap for EGGROLL forward pass."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import jax.lax as lax

from data import prepare_data
from model import init_transformer, count_params
from train_eggroll_optimized import build_param_spec, make_perturbed_forward, winsorized_zscore

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
HALF_POP = 1024
POP_CHUNK = 16
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
SIGMA = 0.04
ALPHA = 0.50
TEMPERATURE = 2.0
SEED = 42


def main():
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]
    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    spec, total_vec_dim = build_param_spec(params)

    train_x = jnp.array(data["train_x"][:BATCH_SIZE])
    train_y = jnp.array(data["train_y"][:BATCH_SIZE])

    key, vec_key = jax.random.split(key)
    vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))

    # ═══════════════════════════════════════════════
    # Benchmark 1: Current JAX vmap approach
    # ═══════════════════════════════════════════════
    forward_fn = make_perturbed_forward(params, config, spec)
    n_chunks = HALF_POP // POP_CHUNK

    def fitness_chunk_fn(carry, chunk_vecs):
        def fitness_pair(vec):
            pos = forward_fn(carry, vec, SIGMA, train_x, train_y, ALPHA)
            neg = forward_fn(carry, vec, -SIGMA, train_x, train_y, ALPHA)
            return pos, neg
        fp, fn = jax.vmap(fitness_pair)(chunk_vecs)
        return carry, (fp, fn)

    jax_scan = jax.jit(lambda p, vc: lax.scan(fitness_chunk_fn, p, vc))
    vc = vecs.reshape(n_chunks, POP_CHUNK, -1)

    print("Benchmarking JAX vmap approach...")
    _, (fp, fn) = jax_scan(params, vc)
    fn.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(3):
        _, (fp, fn) = jax_scan(params, vc)
        fn.block_until_ready()
    t_jax = (time.perf_counter() - t0) / 3
    print(f"  JAX vmap: {t_jax*1000:.0f}ms per round")

    # ═══════════════════════════════════════════════
    # Benchmark 2: Triton kernel
    # ═══════════════════════════════════════════════
    from kernels.fused_transformer_ce import fused_transformer_ce_both

    def run_kernel():
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
            vecs, train_x, train_y,
            SIGMA, ALPHA, TEMPERATURE,
        )

    print("Benchmarking Triton kernel...")
    ce_pos, ce_neg = run_kernel()
    ce_neg.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(3):
        ce_pos, ce_neg = run_kernel()
        ce_neg.block_until_ready()
    t_triton = (time.perf_counter() - t0) / 3
    print(f"  Triton kernel: {t_triton*1000:.0f}ms per round")

    # ═══════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════
    speedup = t_jax / t_triton
    print(f"\nSpeedup: {speedup:.1f}x")
    print(f"\nEstimated training time:")
    print(f"  JAX (current): {t_jax * 2 * 61 * 10:.0f}s")
    print(f"  Triton kernel: {t_triton * 2 * 61 * 10:.0f}s")


if __name__ == "__main__":
    main()
