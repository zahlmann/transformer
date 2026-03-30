"""Test persistent decode: correctness + performance."""

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
from kernels.persistent_decode import persistent_decode


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    PROMPT_LEN = 128
    GEN_LEN = 64

    print(f"Model: d={config['d_model']} h={config['n_heads']} l={config['n_layers']}")
    print()

    # Setup
    prompt = jnp.arange(PROMPT_LEN, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - PROMPT_LEN)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()
    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    # ── Correctness ──
    print("=" * 60)
    print("CORRECTNESS (10 steps)")
    print("=" * 60)

    # Reference: multi_sm per step
    kv_r = kv_packed
    t_r = tok
    ref_tokens = []
    for i in range(10):
        t_r, _, kv_r = multi_sm_decode_nlayer(w, config, t_r, PROMPT_LEN + i, kv_r, vocab_size)
        ref_tokens.append(int(t_r))

    # Persistent decode
    tokens_p, _, _ = persistent_decode(w, config, tok, PROMPT_LEN, kv_packed, vocab_size, 10)

    all_match = True
    for i, (r, p) in enumerate(zip(ref_tokens, tokens_p)):
        match = r == p
        print(f"  Step {i}: ref={r} persistent={p} match={match}")
        if not match:
            all_match = False

    if all_match:
        print("  PASS")
    else:
        print("  FAIL")
        return

    # ── Performance ──
    print()
    print("=" * 60)
    print("PERFORMANCE")
    print("=" * 60)

    # Warmup
    persistent_decode(w, config, tok, PROMPT_LEN, kv_packed, vocab_size, 5)

    # Warmup multi-SM
    kv_tmp = kv_packed
    t = tok
    for i in range(5):
        t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = int(t)

    # Multi-SM per-step (with int sync)
    kv_tmp = kv_packed
    t = tok
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = int(t)
    multi_sync_ms = (time.perf_counter() - t0) * 1000

    # Multi-SM per-step (no int sync)
    kv_tmp = kv_packed
    t = tok
    t0 = time.perf_counter()
    for i in range(GEN_LEN):
        t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
    _ = int(t)
    multi_no_sync_ms = (time.perf_counter() - t0) * 1000

    # Persistent decode (batch alloc + deferred collection)
    t0 = time.perf_counter()
    tokens, _, _ = persistent_decode(w, config, tok, PROMPT_LEN, kv_packed, vocab_size, GEN_LEN)
    persistent_ms = (time.perf_counter() - t0) * 1000

    print(f"  Multi-SM (int sync):     {multi_sync_ms:.1f} ms  ({GEN_LEN/multi_sync_ms*1000:.0f} tok/s)")
    print(f"  Multi-SM (no sync):      {multi_no_sync_ms:.1f} ms  ({GEN_LEN/multi_no_sync_ms*1000:.0f} tok/s)")
    print(f"  Persistent decode:       {persistent_ms:.1f} ms  ({GEN_LEN/persistent_ms*1000:.0f} tok/s)")
    print(f"  Speedup vs sync:         {multi_sync_ms/persistent_ms:.2f}x")


if __name__ == "__main__":
    main()
