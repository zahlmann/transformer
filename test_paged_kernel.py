"""Test GPU-accelerated paged KV cache vs contiguous baseline.

Verifies correctness (0.0 logit diff) and benchmarks throughput.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle, sys, time
import jax.numpy as jnp
from model import prefill_with_kv
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer
from kernels.paged_kv import PagePool


def load_model():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    return params, saved["config"]


def prefill(params, config, prompt_ids):
    ctx_len = config["context_len"]
    prompt_len = len(prompt_ids)
    assert prompt_len < ctx_len
    x = jnp.pad(jnp.array(prompt_ids, dtype=jnp.int32),
                 (0, ctx_len - prompt_len)).astype(jnp.int32)
    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()
    w = prepare_decode_weights_nlayer(params, config, config["vocab_size"])
    kv = pack_kv_caches(k_caches, v_caches)
    first_token = int(jnp.argmax(logits[prompt_len - 1]))
    return w, first_token, prompt_len, kv, k_caches, v_caches


def test_correctness(params, config, prompt_ids, n_steps=32):
    vocab_size = config["vocab_size"]
    max_pages = (config["context_len"] + 63) // 64
    print(f"=== Correctness ({n_steps} steps) ===")

    w, first_token, prompt_len, kv_packed, k_caches, v_caches = prefill(params, config, prompt_ids)

    # warmup contiguous
    tok, _, kv = multi_sm_decode_nlayer(w, config, first_token, prompt_len, kv_packed, vocab_size)
    _ = int(tok)

    # warmup paged
    pp = PagePool(config, max_pages)
    pp.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    pp.sync_to_gpu()
    kv_gpu = pp.to_contiguous_gpu(0)
    tok_g, _, kv_g = multi_sm_decode_nlayer(w, config, first_token, prompt_len, kv_gpu, vocab_size)
    _ = int(tok_g)
    pp.update_page_gpu(0, kv_g, prompt_len)

    # reset for test
    kv_cont = kv_packed
    pp_test = PagePool(config, max_pages)
    pp_test.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    pp_test.sync_to_gpu()
    tok_cont, tok_gpu = first_token, first_token
    max_diff = 0.0
    all_match = True

    for step in range(n_steps):
        pos = prompt_len + step
        tok_c, logits_c, kv_cont = multi_sm_decode_nlayer(w, config, tok_cont, pos, kv_cont, vocab_size)
        tok_c_val = int(tok_c)

        pp_test.ensure_page(0, pos)
        kv_gpu = pp_test.to_contiguous_gpu(0)
        tok_g, logits_g, kv_g = multi_sm_decode_nlayer(w, config, tok_gpu, pos, kv_gpu, vocab_size)
        tok_g_val = int(tok_g)
        pp_test.update_page_gpu(0, kv_g, pos)

        diff = float(jnp.max(jnp.abs(logits_c - logits_g)))
        max_diff = max(max_diff, diff)
        if tok_c_val != tok_g_val:
            print(f"  Step {step}: MISMATCH contiguous={tok_c_val} paged={tok_g_val} diff={diff:.4f}")
            all_match = False
        elif step < 5 or step % 10 == 0:
            print(f"  Step {step}: token={tok_c_val} diff={diff:.6f}")
        tok_cont, tok_gpu = tok_c_val, tok_g_val

    print(f"\nMax diff: {max_diff:.6f}")
    print("PASS" if all_match else "FAIL")
    return all_match


def benchmark(params, config, prompt_ids, n_steps=128):
    vocab_size = config["vocab_size"]
    max_pages = (config["context_len"] + 63) // 64
    print(f"\n=== Benchmark ({n_steps} steps) ===")

    w, first_token, prompt_len, kv_packed, k_caches, v_caches = prefill(params, config, prompt_ids)

    # warmup
    tok, _, kv = multi_sm_decode_nlayer(w, config, first_token, prompt_len, kv_packed, vocab_size)
    _ = int(tok)

    # contiguous
    kv_test, tok_test = kv_packed, first_token
    t0 = time.perf_counter()
    for i in range(n_steps):
        tok_test, _, kv_test = multi_sm_decode_nlayer(w, config, tok_test, prompt_len + i, kv_test, vocab_size)
        _ = int(tok_test)
    t_cont = time.perf_counter() - t0
    print(f"  Contiguous:  {n_steps/t_cont:.0f} tok/s ({t_cont*1000/n_steps:.3f} ms/tok)")

    # gpu paged (with warmup)
    pp = PagePool(config, max_pages)
    pp.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    pp.sync_to_gpu()
    kv_gpu = pp.to_contiguous_gpu(0)
    _, _, kv_g = multi_sm_decode_nlayer(w, config, first_token, prompt_len, kv_gpu, vocab_size)
    pp.update_page_gpu(0, kv_g, prompt_len)

    pp2 = PagePool(config, max_pages)
    pp2.store_prefill_kv(0, k_caches, v_caches, prompt_len)
    pp2.sync_to_gpu()
    tok_test = first_token
    t_gather, t_kernel, t_scatter = 0.0, 0.0, 0.0

    t0 = time.perf_counter()
    for i in range(n_steps):
        pos = prompt_len + i
        pp2.ensure_page(0, pos)
        t1 = time.perf_counter()
        kv_gpu = pp2.to_contiguous_gpu(0)
        t2 = time.perf_counter()
        tok_test, _, kv_out = multi_sm_decode_nlayer(w, config, tok_test, pos, kv_gpu, vocab_size)
        _ = int(tok_test)
        t3 = time.perf_counter()
        pp2.update_page_gpu(0, kv_out, pos)
        t4 = time.perf_counter()
        t_gather += t2 - t1; t_kernel += t3 - t2; t_scatter += t4 - t3
    t_gpu = time.perf_counter() - t0

    print(f"  GPU paged:   {n_steps/t_gpu:.0f} tok/s ({t_gpu*1000/n_steps:.3f} ms/tok)")
    print(f"    gather: {t_gather*1000/n_steps:.3f}ms, kernel: {t_kernel*1000/n_steps:.3f}ms, "
          f"scatter: {t_scatter*1000/n_steps:.3f}ms")
    print(f"  Ratio: {(n_steps/t_gpu)/(n_steps/t_cont):.3f}x")


def main():
    params, config = load_model()
    print(f"d={config['d_model']}, h={config['n_heads']}, l={config['n_layers']}, ctx={config['context_len']}")
    prompt_ids = [1, 42, 100, 7, 255]
    if not test_correctness(params, config, prompt_ids):
        sys.exit(1)
    benchmark(params, config, prompt_ids)


if __name__ == "__main__":
    main()
