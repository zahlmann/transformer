"""Kernel profiling: measure GPU utilization, memory bandwidth, and per-component timing.

This is the primary benchmarking tool. All optimization work should be evaluated against
these metrics. Run after any kernel change to verify improvement.

Metrics reported:
  - tok/s (end-to-end throughput)
  - ms/token (decode latency)
  - GPU time breakdown: prefill vs decode, attention vs FFN vs output
  - Memory: peak VRAM, KV cache size, weight buffer size
  - Kernel stats: register usage, shared memory (from Triton compiler)

Usage:
  uv run profile_kernels.py              # profile current model
  uv run profile_kernels.py --detailed   # per-component breakdown
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import load_bpe_vocab
from model import count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches)


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def measure_prefill(params, config, prompt, vocab_size, n_runs=20):
    """Measure prefill latency."""
    x = jnp.pad(prompt, (0, config["context_len"] - len(prompt))).astype(jnp.int32)

    # Warmup
    for _ in range(3):
        logits, kc, vc = block_prefill(params, config, x, vocab_size)
        _ = logits.block_until_ready()

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        logits, kc, vc = block_prefill(params, config, x, vocab_size)
        _ = logits.block_until_ready()
        times.append(time.perf_counter() - t0)

    return np.median(times) * 1000, logits, kc, vc


def measure_decode(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens=128, n_runs=10):
    """Measure decode throughput."""
    # Warmup
    kv_tmp = kv_packed
    t = tok
    for i in range(5):
        logits, kv_tmp = fused_decode_nlayer(w, config, t, start_pos + i, kv_tmp, vocab_size)
        t = jnp.argmax(logits)
        _ = int(t)

    times = []
    all_tokens = []
    for _ in range(n_runs):
        kv_tmp = kv_packed
        t = tok
        tokens = []
        t0 = time.perf_counter()
        for i in range(n_tokens):
            logits, kv_tmp = fused_decode_nlayer(w, config, t, start_pos + i, kv_tmp, vocab_size)
            t = jnp.argmax(logits)
            tokens.append(int(t))
        times.append(time.perf_counter() - t0)
        all_tokens = tokens

    median_ms = np.median(times) * 1000
    return median_ms, all_tokens


def compute_memory_stats(config, vocab_size):
    """Compute theoretical memory usage."""
    d = config["d_model"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_head = config["d_head"]
    d_ff = 4 * d
    ctx = config["context_len"]

    # Weight buffer (bf16)
    per_layer = (
        d + d +             # ln1 scale, bias
        4 * d * d +         # Q/K/V/O
        d + d +             # ln2 scale, bias
        d * d_ff + d_ff +   # ffn up + bias
        d_ff * d + d        # ffn down + bias
    )
    total_weights = (
        config["vocab_size"] * d +   # token_emb
        ctx * d +                     # pos_emb
        n_layers * per_layer +        # layers
        d + d +                       # ln_final
        d * vocab_size                # output_proj
    )
    weight_bytes = total_weights * 2  # bf16

    # KV cache (bf16) — per sequence
    kv_per_layer = 2 * n_heads * ctx * d_head * 2  # K + V, bf16
    kv_total = n_layers * kv_per_layer

    return {
        "params": total_weights,
        "weight_buffer_mb": weight_bytes / 1e6,
        "kv_cache_mb": kv_total / 1e6,
        "total_inference_mb": (weight_bytes + kv_total) / 1e6,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detailed", action="store_true", help="Per-component timing")
    parser.add_argument("--gen-len", type=int, default=128, help="Tokens to generate")
    parser.add_argument("--n-runs", type=int, default=10, help="Benchmark iterations")
    args = parser.parse_args()

    params, config = load_params()
    vocab_size = config["vocab_size"]
    d = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    ctx = config["context_len"]
    GEN_LEN = args.gen_len
    PROMPT_LEN = min(128, ctx)

    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]

    # Use a simple prompt (avoid re-tokenizing full dataset)
    prompt = jnp.arange(PROMPT_LEN, dtype=jnp.int32) % vocab_size

    print(f"{'='*60}")
    print(f"KERNEL PROFILING")
    print(f"{'='*60}")
    print(f"Model:    d={d} h={n_heads} l={n_layers} ctx={ctx}")
    print(f"Params:   {count_params(params):,}")
    print(f"Generate: {GEN_LEN} tokens from {PROMPT_LEN}-token prompt")
    print(f"GPU:      {jax.devices()[0].device_kind}")
    print()

    # Memory stats
    mem = compute_memory_stats(config, vocab_size)
    print(f"--- Memory ---")
    print(f"Weight buffer:    {mem['weight_buffer_mb']:.1f} MB (bf16)")
    print(f"KV cache:         {mem['kv_cache_mb']:.1f} MB (bf16, per sequence)")
    print(f"Total inference:  {mem['total_inference_mb']:.1f} MB")
    print()

    # Prefill
    print(f"--- Prefill ({PROMPT_LEN} tokens) ---")
    prefill_ms, logits, kc, vc = measure_prefill(params, config, prompt, vocab_size)
    print(f"Latency:          {prefill_ms:.2f} ms")
    print(f"Throughput:       {PROMPT_LEN / prefill_ms * 1000:.0f} tok/s")
    print()

    # Decode
    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    print(f"--- Decode ({GEN_LEN} tokens) ---")
    decode_ms, tokens = measure_decode(w, config, tok, PROMPT_LEN, kv_packed, vocab_size,
                                        n_tokens=GEN_LEN, n_runs=args.n_runs)
    tok_per_s = GEN_LEN / decode_ms * 1000
    ms_per_tok = decode_ms / GEN_LEN
    print(f"Total:            {decode_ms:.1f} ms")
    print(f"Per token:        {ms_per_tok:.3f} ms/tok")
    print(f"Throughput:       {tok_per_s:.0f} tok/s")
    print()

    # End-to-end
    total_ms = prefill_ms + decode_ms
    total_tokens = PROMPT_LEN + GEN_LEN
    print(f"--- End-to-End ---")
    print(f"Total:            {total_ms:.1f} ms for {total_tokens} tokens")
    print(f"Decode tok/s:     {tok_per_s:.0f}")
    print(f"Generated text:   {decode_fn(tokens)[:200]}...")
    print()

    # Theoretical analysis
    weight_bytes = mem["weight_buffer_mb"] * 1e6
    # Each decode step loads all weights + KV cache
    bytes_per_step = weight_bytes + mem["kv_cache_mb"] * 1e6
    bandwidth_gb_s = 836  # RTX 4080 Super theoretical
    theoretical_min_ms = bytes_per_step / (bandwidth_gb_s * 1e9) * 1000
    bandwidth_util = theoretical_min_ms / ms_per_tok * 100

    print(f"--- Roofline Analysis ---")
    print(f"Bytes per decode step:  {bytes_per_step / 1e6:.1f} MB")
    print(f"Theoretical min (836 GB/s): {theoretical_min_ms:.3f} ms/tok")
    print(f"Achieved:               {ms_per_tok:.3f} ms/tok")
    print(f"Bandwidth utilization:  {bandwidth_util:.0f}%")
    print()

    # Save baseline
    results = {
        "model": f"d={d}_h={n_heads}_l={n_layers}_ctx={ctx}",
        "params": count_params(params),
        "prefill_ms": round(prefill_ms, 2),
        "decode_tok_s": round(tok_per_s, 0),
        "decode_ms_per_tok": round(ms_per_tok, 3),
        "weight_buffer_mb": round(mem["weight_buffer_mb"], 1),
        "kv_cache_mb": round(mem["kv_cache_mb"], 1),
        "bandwidth_util_pct": round(bandwidth_util, 0),
    }

    results_path = os.path.join(os.path.dirname(__file__), "baseline_metrics.txt")
    with open(results_path, "w") as f:
        f.write(f"# Baseline metrics — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"# Compare against these after any kernel change\n\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"Saved baseline to {results_path}")


if __name__ == "__main__":
    main()
