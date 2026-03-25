"""Profile EGGROLL training to find bottlenecks."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from data import prepare_data
from model import init_transformer, count_params

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
HALF_POP = 1024
POP_CHUNK = 16
SIGMA = 0.04
LR = 0.012
ALPHA = 0.50
TEMPERATURE = 2.0
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
SEED = 42


def build_param_spec(params):
    spec = []
    offset = 0
    for key in sorted(params.keys()):
        shape = params[key].shape
        if len(shape) == 2:
            vec_dim = shape[0] + shape[1]
            spec.append((key, shape, offset, vec_dim, True))
        elif len(shape) == 1:
            vec_dim = shape[0]
            spec.append((key, shape, offset, vec_dim, False))
        else:
            raise ValueError(f"Unexpected: {key} {shape}")
        offset += vec_dim
    return spec, offset


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
    n_chunks = HALF_POP // POP_CHUNK

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    x = train_x[:BATCH_SIZE]
    y = train_y[:BATCH_SIZE]

    # Import the forward function from optimized
    from train_eggroll_optimized import make_perturbed_forward, winsorized_zscore
    forward_fn = make_perturbed_forward(params, config, spec)

    # --- Profile: QR decomposition ---
    @jax.jit
    def do_qr(key):
        raw = jax.random.normal(key, (total_vec_dim, HALF_POP))
        Q, _ = jnp.linalg.qr(raw)
        vecs = Q.T * jnp.sqrt(jnp.float32(total_vec_dim))
        return vecs

    print("Profiling QR decomposition...")
    key, k = jax.random.split(key)
    vecs = do_qr(k)
    vecs.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        key, k = jax.random.split(key)
        vecs = do_qr(k)
        vecs.block_until_ready()
    t_qr = (time.perf_counter() - t0) / 10
    print(f"  QR decomp: {t_qr*1000:.1f}ms per call")
    print(f"  Per epoch (61 batches × 2 accum = 122 calls): {t_qr*122:.2f}s")

    # --- Profile: random normal (no QR) ---
    @jax.jit
    def do_random(key):
        vecs = jax.random.normal(key, (HALF_POP, total_vec_dim))
        return vecs

    key, k = jax.random.split(key)
    vecs_r = do_random(k)
    vecs_r.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(10):
        key, k = jax.random.split(key)
        vecs_r = do_random(k)
        vecs_r.block_until_ready()
    t_rand = (time.perf_counter() - t0) / 10
    print(f"  Random normal (no QR): {t_rand*1000:.1f}ms per call")

    # --- Profile: vmapped forward pass (one chunk of 16) ---
    def fitness_pair(vec):
        pos = forward_fn(params, vec, SIGMA, x, y, ALPHA)
        neg = forward_fn(params, vec, -SIGMA, x, y, ALPHA)
        return pos, neg

    fitness_pair_vmap = jax.jit(jax.vmap(fitness_pair))

    print("\nProfiling forward pass (chunk of 16)...")
    chunk = vecs[:POP_CHUNK]
    fp, fn = fitness_pair_vmap(chunk)
    fp.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(20):
        fp, fn = fitness_pair_vmap(chunk)
        fn.block_until_ready()
    t_chunk = (time.perf_counter() - t0) / 20
    print(f"  Forward chunk (16 pairs = 32 passes): {t_chunk*1000:.1f}ms")
    print(f"  Per forward pass: {t_chunk/32*1000:.2f}ms")
    print(f"  All chunks per round ({n_chunks}): {t_chunk*n_chunks*1000:.0f}ms")
    print(f"  Per epoch (61 batches × 2 rounds): {t_chunk*n_chunks*2*61:.2f}s")

    # --- Profile: lax.scan over chunks ---
    def fitness_chunk_fn(carry, chunk_vecs):
        def fitness_pair_inner(vec):
            pos = forward_fn(carry, vec, SIGMA, x, y, ALPHA)
            neg = forward_fn(carry, vec, -SIGMA, x, y, ALPHA)
            return pos, neg
        fp, fn = jax.vmap(fitness_pair_inner)(chunk_vecs)
        return carry, (fp, fn)

    scan_fn = jax.jit(lambda p, vc: lax.scan(fitness_chunk_fn, p, vc))

    print("\nProfiling lax.scan over all chunks...")
    vc = vecs.reshape(n_chunks, POP_CHUNK, -1)
    _, (all_fp, all_fn) = scan_fn(params, vc)
    all_fn.block_until_ready()
    t0 = time.perf_counter()
    for _ in range(5):
        _, (all_fp, all_fn) = scan_fn(params, vc)
        all_fn.block_until_ready()
    t_scan = (time.perf_counter() - t0) / 5
    print(f"  lax.scan all chunks: {t_scan*1000:.0f}ms")
    print(f"  Overhead vs sum of chunks: {(t_scan - t_chunk*n_chunks)*1000:.0f}ms")

    # --- Profile: gradient computation ---
    @jax.jit
    def compute_grad(vecs, diffs):
        shaped = winsorized_zscore(diffs)
        scale = 1.0 / (2.0 * SIGMA * HALF_POP)
        grads = {}
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v = vecs[:, offset:offset + vec_dim]
            if is_2d:
                m, n = shape
                grads[pkey] = scale * (v[:, :m] * shaped[:, None]).T @ v[:, m:]
            else:
                grads[pkey] = scale * (v * shaped[:, None]).sum(axis=0)
        return grads

    print("\nProfiling gradient computation...")
    diffs = all_fp.reshape(-1) - all_fn.reshape(-1)
    grads = compute_grad(vecs, diffs)
    jax.block_until_ready(grads)
    t0 = time.perf_counter()
    for _ in range(20):
        grads = compute_grad(vecs, diffs)
        jax.block_until_ready(grads)
    t_grad = (time.perf_counter() - t0) / 20
    print(f"  Gradient computation: {t_grad*1000:.1f}ms")
    print(f"  Per epoch: {t_grad*2*61:.2f}s")

    # --- Profile: different POP_CHUNK sizes ---
    print("\nProfiling different POP_CHUNK sizes...")
    for pc in [8, 16, 32, 64]:
        nc = HALF_POP // pc
        vc_test = vecs.reshape(nc, pc, -1)
        scan_fn_test = jax.jit(lambda p, vc: lax.scan(
            lambda carry, chunk_vecs: (carry, jax.vmap(
                lambda vec: (forward_fn(carry, vec, SIGMA, x, y, ALPHA),
                             forward_fn(carry, vec, -SIGMA, x, y, ALPHA))
            )(chunk_vecs)),
            p, vc))
        # warmup
        _, results = scan_fn_test(params, vc_test)
        jax.block_until_ready(results)
        t0 = time.perf_counter()
        for _ in range(3):
            _, results = scan_fn_test(params, vc_test)
            jax.block_until_ready(results)
        t = (time.perf_counter() - t0) / 3
        print(f"  POP_CHUNK={pc}: {t*1000:.0f}ms per round ({nc} chunks)")

    # --- Summary ---
    print("\n=== SUMMARY (estimated per epoch) ===")
    total_per_epoch = t_qr * 122 + t_scan * 2 * 61 + t_grad * 2 * 61
    print(f"  QR:        {t_qr*122:.1f}s ({t_qr*122/total_per_epoch*100:.0f}%)")
    print(f"  Forward:   {t_scan*2*61:.1f}s ({t_scan*2*61/total_per_epoch*100:.0f}%)")
    print(f"  Gradient:  {t_grad*2*61:.1f}s ({t_grad*2*61/total_per_epoch*100:.0f}%)")
    print(f"  Total est: {total_per_epoch:.1f}s per epoch, {total_per_epoch*10:.1f}s for 10 epochs")


if __name__ == "__main__":
    main()
