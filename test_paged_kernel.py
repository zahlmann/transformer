"""Test GPU-accelerated paged KV cache vs Python-level paging vs contiguous.

Benchmarks three approaches:
  1. Contiguous: standard decode with contiguous KV buffer (baseline)
  2. Python-level paging: to_contiguous() / update_from_contiguous() with CPU numpy
  3. GPU paging: to_contiguous_gpu() / update_page_gpu() with Triton copy kernels
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from model import prefill_with_kv
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer
from kernels.paged_kv import PagePool


def load_model(weights_path="weights.pkl"):
    with open(os.path.join(os.path.dirname(__file__), weights_path), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def prefill_and_setup(params, config, prompt_ids):
    ctx_len = config["context_len"]
    prompt_len = len(prompt_ids)
    vocab_size = config["vocab_size"]

    x = jnp.pad(jnp.array(prompt_ids, dtype=jnp.int32),
                 (0, ctx_len - prompt_len)).astype(jnp.int32)
    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()

    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    kv_packed = pack_kv_caches(k_caches, v_caches)
    first_token = int(jnp.argmax(logits[prompt_len - 1]))

    return w, first_token, prompt_len, kv_packed, k_caches, v_caches


def test_correctness(params, config, prompt_ids, n_steps=32):
    """Verify GPU-paged path matches contiguous path."""
    vocab_size = config["vocab_size"]
    print(f"=== Correctness Test (d={config['d_model']}, "
          f"l={config['n_layers']}, h={config['n_heads']}) ===")

    w, first_token, prompt_len, kv_packed, k_caches, v_caches = \
        prefill_and_setup(params, config, prompt_ids)

    # Set up paged pool
    max_pages = (config["context_len"] + 63) // 64
    page_pool = PagePool(config, max_pages=max_pages, page_size=64)
    page_pool.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    page_pool.sync_to_gpu()

    # Warmup both paths
    print("Warming up contiguous kernel...", end=" ", flush=True)
    tok_c, _, kv_c = multi_sm_decode_nlayer(
        w, config, first_token, prompt_len, kv_packed, vocab_size)
    _ = int(tok_c)
    print("done")

    print("Warming up GPU copy kernels...", end=" ", flush=True)
    kv_gpu = page_pool.to_contiguous_gpu(0)
    _ = kv_gpu.block_until_ready()
    tok_g, _, kv_g_out = multi_sm_decode_nlayer(
        w, config, first_token, prompt_len, kv_gpu, vocab_size)
    _ = int(tok_g)
    page_pool.update_page_gpu(0, kv_g_out, prompt_len)
    _ = page_pool.gpu_pool.block_until_ready()
    print("done")

    # Reset for test
    kv_cont = kv_packed
    page_pool_test = PagePool(config, max_pages=max_pages, page_size=64)
    page_pool_test.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    page_pool_test.sync_to_gpu()

    tok_cont = first_token
    tok_gpu = first_token
    max_logit_diff = 0.0
    all_match = True

    for step in range(n_steps):
        pos = prompt_len + step

        # Contiguous decode
        tok_c, logits_c, kv_cont = multi_sm_decode_nlayer(
            w, config, tok_cont, pos, kv_cont, vocab_size)
        tok_c_val = int(tok_c)

        # GPU-paged decode
        page_pool_test.ensure_page_for_pos(0, pos)
        kv_gpu = page_pool_test.to_contiguous_gpu(0)
        tok_g, logits_g, kv_g_out = multi_sm_decode_nlayer(
            w, config, tok_gpu, pos, kv_gpu, vocab_size)
        tok_g_val = int(tok_g)
        page_pool_test.update_page_gpu(0, kv_g_out, pos)

        logit_diff = float(jnp.max(jnp.abs(logits_c - logits_g)))
        max_logit_diff = max(max_logit_diff, logit_diff)

        if tok_c_val != tok_g_val:
            print(f"  Step {step}: TOKEN MISMATCH! contiguous={tok_c_val} gpu_paged={tok_g_val} "
                  f"(logit diff={logit_diff:.4f})")
            all_match = False
        elif step < 5 or step % 10 == 0:
            print(f"  Step {step}: token={tok_c_val} logit_diff={logit_diff:.6f}")

        tok_cont = tok_c_val
        tok_gpu = tok_g_val

    print(f"\nMax logit diff: {max_logit_diff:.6f}")
    if all_match:
        print("PASS: All tokens match!")
    else:
        print("FAIL: Token mismatches detected")
    return all_match


def benchmark(params, config, prompt_ids, n_steps=128):
    """Benchmark contiguous vs Python-paged vs GPU-paged."""
    vocab_size = config["vocab_size"]
    n_layers = config["n_layers"]
    n_kv_heads = config.get("n_kv_heads", config["n_heads"])
    d_head = config["d_head"]
    max_seq = config["context_len"]
    max_pages = (max_seq + 63) // 64

    print(f"\n=== Benchmark ({n_steps} steps) ===")

    w, first_token, prompt_len, kv_packed, k_caches, v_caches = \
        prefill_and_setup(params, config, prompt_ids)

    # Warmup
    tok, _, kv = multi_sm_decode_nlayer(
        w, config, first_token, prompt_len, kv_packed, vocab_size)
    _ = int(tok)

    # 1. Contiguous baseline (with per-token sync)
    kv_test = kv_packed; tok_test = first_token
    t0 = time.perf_counter()
    for i in range(n_steps):
        tok_test, _, kv_test = multi_sm_decode_nlayer(
            w, config, tok_test, prompt_len + i, kv_test, vocab_size)
        _ = int(tok_test)
    t_cont = time.perf_counter() - t0
    cont_tps = n_steps / t_cont
    print(f"  Contiguous:     {cont_tps:.0f} tok/s ({t_cont*1000/n_steps:.3f} ms/tok)")

    # 2. Python-level paging
    pp_py = PagePool(config, max_pages=max_pages, page_size=64)
    pp_py.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    tok_test = first_token

    t0 = time.perf_counter()
    for i in range(n_steps):
        pos = prompt_len + i
        pp_py.ensure_page_for_pos(0, pos)
        kv_c = pp_py.to_contiguous(0)
        tok_test, _, kv_out = multi_sm_decode_nlayer(
            w, config, tok_test, pos, kv_c, vocab_size)
        _ = int(tok_test)
        pp_py.update_from_contiguous(0, kv_out, pos)
    t_py = time.perf_counter() - t0
    py_tps = n_steps / t_py
    print(f"  Python paged:   {py_tps:.0f} tok/s ({t_py*1000/n_steps:.3f} ms/tok)")

    # 3. GPU-accelerated paging
    pp_gpu = PagePool(config, max_pages=max_pages, page_size=64)
    pp_gpu.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    pp_gpu.sync_to_gpu()

    # Warmup GPU copy kernels
    kv_gpu = pp_gpu.to_contiguous_gpu(0)
    _ = kv_gpu.block_until_ready()
    _, _, kv_gpu_out = multi_sm_decode_nlayer(
        w, config, first_token, prompt_len, kv_gpu, vocab_size)
    pp_gpu.update_page_gpu(0, kv_gpu_out, prompt_len)
    _ = pp_gpu.gpu_pool.block_until_ready()

    # Reset
    pp_gpu2 = PagePool(config, max_pages=max_pages, page_size=64)
    pp_gpu2.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    pp_gpu2.sync_to_gpu()
    tok_test = first_token

    t_to_cont = 0.0; t_kernel = 0.0; t_update = 0.0
    t0 = time.perf_counter()
    for i in range(n_steps):
        pos = prompt_len + i
        pp_gpu2.ensure_page_for_pos(0, pos)

        t1 = time.perf_counter()
        kv_gpu = pp_gpu2.to_contiguous_gpu(0)
        t2 = time.perf_counter()
        t_to_cont += t2 - t1

        tok_test, _, kv_out = multi_sm_decode_nlayer(
            w, config, tok_test, pos, kv_gpu, vocab_size)
        _ = int(tok_test)
        t3 = time.perf_counter()
        t_kernel += t3 - t2

        pp_gpu2.update_page_gpu(0, kv_out, pos)
        t4 = time.perf_counter()
        t_update += t4 - t3

    t_gpu = time.perf_counter() - t0
    gpu_tps = n_steps / t_gpu
    print(f"  GPU paged:      {gpu_tps:.0f} tok/s ({t_gpu*1000/n_steps:.3f} ms/tok)")

    print(f"\n  GPU paged breakdown per step:")
    print(f"    to_contiguous_gpu:  {t_to_cont*1000/n_steps:.3f} ms")
    print(f"    decode kernel:      {t_kernel*1000/n_steps:.3f} ms")
    print(f"    update_page_gpu:    {t_update*1000/n_steps:.3f} ms")

    print(f"\n  Speedup vs Python paged: {gpu_tps/py_tps:.2f}x")
    print(f"  Overhead vs contiguous:  {gpu_tps/cont_tps:.3f}x")

    return cont_tps, py_tps, gpu_tps


def main():
    print("Loading model...")
    params, config = load_model()
    print(f"Model: d={config['d_model']}, h={config['n_heads']}, "
          f"l={config['n_layers']}, ctx={config['context_len']}")

    prompt_ids = [1, 42, 100, 7, 255]

    passed = test_correctness(params, config, prompt_ids, n_steps=32)
    if not passed:
        print("\nCorrectness test failed, skipping benchmark")
        sys.exit(1)

    benchmark(params, config, prompt_ids, n_steps=128)


if __name__ == "__main__":
    main()
