"""Measure decode throughput with and without GPU->CPU sync per token."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_params():
    with open("weights.pkl", "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    prompt_len = 128
    gen_len = 128
    n_runs = 10

    prompt = jnp.arange(prompt_len, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - prompt_len)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[prompt_len - 1])

    # Warmup
    kv = kv_packed
    t = tok
    for i in range(5):
        t, _, kv = multi_sm_decode_nlayer(w, config, t, prompt_len + i, kv, vocab_size)
        _ = int(t)

    # With int() sync per token
    times_sync = []
    for _ in range(n_runs):
        kv = kv_packed
        t = tok
        t0 = time.perf_counter()
        for i in range(gen_len):
            t, _, kv = multi_sm_decode_nlayer(w, config, t, prompt_len + i, kv, vocab_size)
            _ = int(t)
        times_sync.append(time.perf_counter() - t0)

    # Without sync: use next_token directly (stays on device)
    times_nosync = []
    for _ in range(n_runs):
        kv = kv_packed
        t = tok
        t0 = time.perf_counter()
        for i in range(gen_len):
            t, _, kv = multi_sm_decode_nlayer(w, config, t, prompt_len + i, kv, vocab_size)
        # Single sync at the end
        _ = int(t)
        times_nosync.append(time.perf_counter() - t0)

    # block_until_ready (no int, just GPU sync)
    times_block = []
    for _ in range(n_runs):
        kv = kv_packed
        t = tok
        t0 = time.perf_counter()
        for i in range(gen_len):
            t, logits, kv = multi_sm_decode_nlayer(w, config, t, prompt_len + i, kv, vocab_size)
        _ = kv.block_until_ready()
        times_block.append(time.perf_counter() - t0)

    ms_sync = np.median(times_sync) * 1000
    ms_nosync = np.median(times_nosync) * 1000
    ms_block = np.median(times_block) * 1000

    print(f"=== Decode {gen_len} tokens (kv_splits=2, grid=32) ===")
    print(f"With int() sync:    {ms_sync:.1f} ms  ({gen_len/ms_sync*1000:.0f} tok/s)  ({ms_sync/gen_len:.3f} ms/tok)")
    print(f"No per-tok sync:    {ms_nosync:.1f} ms  ({gen_len/ms_nosync*1000:.0f} tok/s)  ({ms_nosync/gen_len:.3f} ms/tok)")
    print(f"block_until_ready:  {ms_block:.1f} ms  ({gen_len/ms_block*1000:.0f} tok/s)  ({ms_block/gen_len:.3f} ms/tok)")
    print()
    sync_overhead = (ms_sync - ms_nosync) / gen_len
    print(f"Per-token sync overhead: {sync_overhead*1000:.0f} µs")
    print(f"Sync fraction of total: {(ms_sync - ms_nosync) / ms_sync * 100:.0f}%")

    # Save results
    with open("nosync_results.txt", "w") as f:
        f.write(f"with_sync_tok_s: {gen_len/ms_sync*1000:.0f}\n")
        f.write(f"no_sync_tok_s: {gen_len/ms_nosync*1000:.0f}\n")
        f.write(f"block_tok_s: {gen_len/ms_block*1000:.0f}\n")
        f.write(f"sync_overhead_us: {sync_overhead*1000:.0f}\n")


if __name__ == "__main__":
    main()
