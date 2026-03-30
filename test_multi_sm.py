"""Test multi-SM decode kernel: correctness + performance comparison."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from model import count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches)
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    d = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    ctx = config["context_len"]
    PROMPT_LEN = 128
    GEN_LEN = 64

    print(f"Model: d={d} h={n_heads} l={n_layers} ctx={ctx}")
    print(f"Params: {count_params(params):,}")
    print()

    # Prefill
    prompt = jnp.arange(PROMPT_LEN, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, ctx - PROMPT_LEN)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    # ── Correctness test ──
    print("=" * 60)
    print("CORRECTNESS TEST")
    print("=" * 60)

    # Single-SM decode (reference)
    logits_ref, kv_ref = fused_decode_nlayer(w, config, tok, PROMPT_LEN, kv_packed, vocab_size)
    _ = logits_ref.block_until_ready()

    # Multi-SM decode (workspace as outputs)
    logits_new, kv_new = multi_sm_decode_nlayer(w, config, tok, PROMPT_LEN, kv_packed, vocab_size)
    _ = logits_new.block_until_ready()

    max_diff = float(jnp.max(jnp.abs(logits_ref - logits_new)))
    mean_diff = float(jnp.mean(jnp.abs(logits_ref - logits_new)))
    ref_tok = int(jnp.argmax(logits_ref))
    new_tok = int(jnp.argmax(logits_new))

    print(f"  Max logit diff:     {max_diff:.6f}")
    print(f"  Mean logit diff:    {mean_diff:.6f}")
    print(f"  Top-1 match:        {ref_tok == new_tok} (ref={ref_tok}, new={new_tok})")

    if max_diff < 0.1 and ref_tok == new_tok:
        print("  PASS")
    else:
        print("  FAIL")
        return

    # Multi-step
    print()
    print("Multi-step correctness (10 steps):")
    kv_r = kv_packed
    kv_n = kv_packed
    t_r = tok
    t_n = tok
    all_match = True
    for i in range(10):
        logits_r, kv_r = fused_decode_nlayer(w, config, t_r, PROMPT_LEN + i, kv_r, vocab_size)
        logits_n, kv_n = multi_sm_decode_nlayer(w, config, t_n, PROMPT_LEN + i, kv_n, vocab_size)
        t_r = jnp.argmax(logits_r)
        t_n = jnp.argmax(logits_n)
        match = int(t_r) == int(t_n)
        diff = float(jnp.max(jnp.abs(logits_r - logits_n)))
        print(f"  Step {i}: ref={int(t_r)} new={int(t_n)} match={match} max_diff={diff:.4f}")
        if not match:
            all_match = False

    if all_match:
        print("  ALL STEPS MATCH")
    else:
        print("  MISMATCH DETECTED")
        return

    # ── Performance test ──
    print()
    print("=" * 60)
    print("PERFORMANCE TEST")
    print("=" * 60)

    # Warmup
    kv_tmp = kv_packed
    t = tok
    for i in range(5):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)

    # Single-SM
    kv_tmp = kv_packed
    t = tok
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = fused_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)
    single_ms = (time.perf_counter() - t0) * 1000

    # Warmup multi-SM
    kv_tmp = kv_packed
    t = tok
    for i in range(5):
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)

    # Multi-SM with int() sync
    kv_tmp = kv_packed
    t = tok
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
        _ = int(t)
    multi_sync_ms = (time.perf_counter() - t0) * 1000

    # Multi-SM without int() sync
    kv_tmp = kv_packed
    t = tok
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        logits_d, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits_d)
    _ = int(t)
    multi_no_sync_ms = (time.perf_counter() - t0) * 1000

    print(f"  Single-SM (int sync):    {single_ms:.1f} ms  ({GEN_LEN/single_ms*1000:.0f} tok/s)")
    print(f"  Multi-SM (int sync):     {multi_sync_ms:.1f} ms  ({GEN_LEN/multi_sync_ms*1000:.0f} tok/s)")
    print(f"  Multi-SM (no int sync):  {multi_no_sync_ms:.1f} ms  ({GEN_LEN/multi_no_sync_ms*1000:.0f} tok/s)")
    print(f"  Speedup (sync):          {single_ms/multi_sync_ms:.2f}x")
    print(f"  Speedup (no sync):       {single_ms/multi_no_sync_ms:.2f}x")

    with open(os.path.join(os.path.dirname(__file__), "multi_sm_results.txt"), "w") as f:
        f.write(f"# Multi-SM decode test — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"single_sm_ms: {single_ms:.1f}\n")
        f.write(f"multi_sm_sync_ms: {multi_sync_ms:.1f}\n")
        f.write(f"multi_sm_no_sync_ms: {multi_no_sync_ms:.1f}\n")
        f.write(f"speedup_sync: {single_ms/multi_sync_ms:.2f}\n")
        f.write(f"speedup_no_sync: {single_ms/multi_no_sync_ms:.2f}\n")
        f.write(f"multi_sync_tok_s: {GEN_LEN/multi_sync_ms*1000:.0f}\n")
        f.write(f"multi_no_sync_tok_s: {GEN_LEN/multi_no_sync_ms*1000:.0f}\n")


if __name__ == "__main__":
    main()
