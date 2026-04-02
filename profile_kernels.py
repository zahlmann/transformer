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
from model import prefill_with_kv
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches)
from kernels.multi_sm_decode import multi_sm_decode_nlayer
from kernels.batched_decode import batched_decode_nlayer
from kernels.persistent_decode import persistent_decode_nlayer
from kernels.persistent_batched_decode import persistent_batched_decode_nlayer


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def measure_prefill(params, config, prompt, vocab_size, n_runs=20):
    """Measure prefill latency. Uses Triton prefill for all models (including GQA)."""
    x = jnp.pad(prompt, (0, config["context_len"] - len(prompt))).astype(jnp.int32)

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


def measure_decode(w, config, tok, start_pos, kv_packed, vocab_size, n_tokens=128, n_runs=10,
                   use_multi_sm=True):
    """Measure decode throughput."""
    if use_multi_sm:
        # Multi-SM returns (next_token, logits, kv_out)
        # Use in-kernel argmax — no jnp.argmax needed
        # Warmup
        kv_tmp = kv_packed
        t = tok
        for i in range(5):
            t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv_tmp, vocab_size)
            _ = int(t)

        times = []
        all_tokens = []
        for _ in range(n_runs):
            kv_tmp = kv_packed
            t = tok
            tokens = []
            t0 = time.perf_counter()
            for i in range(n_tokens):
                t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, start_pos + i, kv_tmp, vocab_size)
                tokens.append(int(t))
            times.append(time.perf_counter() - t0)
            all_tokens = tokens
    else:
        # Single-SM returns (logits, kv_out)
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


def measure_batched_decode(w, config, tok, start_pos, kv_packed, vocab_size,
                           batch_size=4, n_tokens=128, n_runs=10):
    """Measure batched decode throughput.

    Runs B identical sequences in parallel to isolate the kernel speedup.
    Total throughput = B * n_tokens / time.
    """
    # Replicate inputs for B sequences
    token_ids = jnp.full((batch_size,), tok, dtype=jnp.int32)
    positions = jnp.full((batch_size,), start_pos, dtype=jnp.int32)
    # Concatenate B copies of the KV cache
    kv_batch = jnp.tile(kv_packed, batch_size)

    # Warmup
    kv_tmp = kv_batch
    tids = token_ids
    for i in range(5):
        pos = jnp.full((batch_size,), start_pos + i, dtype=jnp.int32)
        next_toks, _, kv_tmp = batched_decode_nlayer(
            w, config, tids, pos, kv_tmp, vocab_size, batch_size)
        tids = next_toks
        _ = int(next_toks[0])

    times = []
    all_tokens = []
    for _ in range(n_runs):
        kv_tmp = kv_batch
        tids = token_ids
        tokens_b0 = []
        t0 = time.perf_counter()
        for i in range(n_tokens):
            pos = jnp.full((batch_size,), start_pos + i, dtype=jnp.int32)
            next_toks, _, kv_tmp = batched_decode_nlayer(
                w, config, tids, pos, kv_tmp, vocab_size, batch_size)
            tids = next_toks
            tokens_b0.append(int(next_toks[0]))
        times.append(time.perf_counter() - t0)
        all_tokens = tokens_b0

    median_ms = np.median(times) * 1000
    return median_ms, all_tokens


def measure_batched_pipelined(w, config, tok, start_pos, kv_packed, vocab_size,
                              batch_size=4, n_tokens=128, n_runs=10):
    """Measure pipelined batched decode (no per-step int() sync)."""
    token_ids = jnp.full((batch_size,), tok, dtype=jnp.int32)
    kv_batch = jnp.tile(kv_packed, batch_size)

    # Warmup
    kv_tmp = kv_batch
    tids = token_ids
    for i in range(5):
        pos = jnp.full((batch_size,), start_pos + i, dtype=jnp.int32)
        tids, _, kv_tmp = batched_decode_nlayer(w, config, tids, pos, kv_tmp, vocab_size, batch_size)
    _ = tids.block_until_ready()

    times = []
    all_tokens = []
    for _ in range(n_runs):
        kv_tmp = kv_batch
        tids = token_ids
        tok_devs = []
        t0 = time.perf_counter()
        for i in range(n_tokens):
            pos = jnp.full((batch_size,), start_pos + i, dtype=jnp.int32)
            tids, _, kv_tmp = batched_decode_nlayer(w, config, tids, pos, kv_tmp, vocab_size, batch_size)
            tok_devs.append(tids[0])
        pipe_toks = [int(td) for td in tok_devs]
        times.append(time.perf_counter() - t0)
        all_tokens = pipe_toks

    return np.median(times) * 1000, all_tokens


def compute_memory_stats(config, vocab_size):
    """Compute theoretical memory usage."""
    d = config["d_model"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    n_layers = config["n_layers"]
    d_head = config["d_head"]
    d_kv = n_kv_heads * d_head
    d_ff = 4 * d
    ctx = config["context_len"]

    # Weight buffer (bf16) — Q/O use d_model, K/V use d_kv (GQA)
    per_layer = (
        d + d +             # ln1 scale, bias
        d * d +             # Q
        d * d_kv +          # K (GQA: smaller)
        d * d_kv +          # V (GQA: smaller)
        d * d +             # O
        d + d +             # ln2 scale, bias
        d * d_ff +          # ffn up
        d_ff * d            # ffn down
    )
    total_weights = (
        config["vocab_size"] * d +   # token_emb
        ctx * d +                     # pos_emb
        n_layers * per_layer +        # layers
        d + d +                       # ln_final
        d * vocab_size                # output_proj
    )
    weight_bytes = total_weights * 2  # bf16

    # KV cache (bf16) — per sequence (GQA: uses n_kv_heads)
    kv_per_layer = 2 * n_kv_heads * ctx * d_head * 2  # K + V, bf16
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

    # Multi-SM decode (primary) — kv_splits=2 gives grid=N_HEADS*2
    kv_splits = 2
    total_blocks = n_heads * kv_splits
    print(f"--- Decode: Multi-SM ({GEN_LEN} tokens, grid={total_blocks}, kv_splits={kv_splits}) ---")
    decode_ms, tokens = measure_decode(w, config, tok, PROMPT_LEN, kv_packed, vocab_size,
                                        n_tokens=GEN_LEN, n_runs=args.n_runs, use_multi_sm=True)
    tok_per_s = GEN_LEN / decode_ms * 1000
    ms_per_tok = decode_ms / GEN_LEN
    print(f"Total:            {decode_ms:.1f} ms")
    print(f"Per token:        {ms_per_tok:.3f} ms/tok")
    print(f"Throughput:       {tok_per_s:.0f} tok/s")
    print()

    # Single-SM decode (reference) — skip for GQA models (single-SM kernel doesn't support GQA)
    is_gqa = config.get("n_kv_heads", n_heads) != n_heads
    if not is_gqa:
        print(f"--- Decode: Single-SM ({GEN_LEN} tokens, grid=1) ---")
        decode_ms_old, _ = measure_decode(w, config, tok, PROMPT_LEN, kv_packed, vocab_size,
                                           n_tokens=GEN_LEN, n_runs=args.n_runs, use_multi_sm=False)
        tok_per_s_old = GEN_LEN / decode_ms_old * 1000
        ms_per_tok_old = decode_ms_old / GEN_LEN
        print(f"Total:            {decode_ms_old:.1f} ms")
        print(f"Per token:        {ms_per_tok_old:.3f} ms/tok")
        print(f"Throughput:       {tok_per_s_old:.0f} tok/s")
        print(f"Multi-SM speedup: {decode_ms_old / decode_ms:.2f}x")
        print()
    else:
        decode_ms_old = decode_ms
        tok_per_s_old = tok_per_s
        print(f"--- Decode: Single-SM skipped (GQA not supported) ---")
        print()

    # Batched decode
    for B in [4, 8, 16]:
        print(f"--- Decode: Batched B={B} ({GEN_LEN} tokens/seq, grid={total_blocks}) ---")
        try:
            batch_ms, batch_tokens = measure_batched_decode(
                w, config, tok, PROMPT_LEN, kv_packed, vocab_size,
                batch_size=B, n_tokens=GEN_LEN, n_runs=args.n_runs)
            total_tokens_batch = B * GEN_LEN
            batch_tok_per_s = total_tokens_batch / batch_ms * 1000
            batch_ms_per_tok = batch_ms / GEN_LEN
            print(f"Total:            {batch_ms:.1f} ms ({GEN_LEN} steps)")
            print(f"Per step:         {batch_ms_per_tok:.3f} ms")
            print(f"Throughput:       {batch_tok_per_s:.0f} tok/s total ({batch_tok_per_s/B:.0f} per seq)")
            print(f"Speedup vs B=1:   {batch_tok_per_s / tok_per_s:.2f}x total throughput")
            print(f"Text (seq 0):     {decode_fn(batch_tokens)[:100]}...")
            # Verify seq 0 matches single-sequence
            match = batch_tokens == tokens
            print(f"Matches single:   {'YES' if match else 'NO (expected for greedy)'}")
        except Exception as e:
            print(f"ERROR: {e}")
        print()

    # Pipelined batched decode (no per-token int() sync)
    for B in [4, 8]:
        print(f"--- Batched Pipelined B={B} ({GEN_LEN} tokens/seq, no per-step sync) ---")
        try:
            pipe_batch_ms, pipe_batch_tokens = measure_batched_pipelined(
                w, config, tok, PROMPT_LEN, kv_packed, vocab_size,
                batch_size=B, n_tokens=GEN_LEN, n_runs=args.n_runs)
            pipe_batch_tok_s = B * GEN_LEN / pipe_batch_ms * 1000
            print(f"Total:            {pipe_batch_ms:.1f} ms ({GEN_LEN} steps)")
            print(f"Per step:         {pipe_batch_ms/GEN_LEN:.3f} ms")
            print(f"Throughput:       {pipe_batch_tok_s:.0f} tok/s total ({pipe_batch_tok_s/B:.0f} per seq)")
        except Exception as e:
            print(f"ERROR: {e}")
        print()

    # Persistent batched decode (single kernel launch for all steps × B sequences)
    for B in [4, 8]:
        print(f"--- Persistent Batched B={B} ({GEN_LEN} tokens/seq, single launch) ---")
        try:
            first_tokens = jnp.full((B,), tok, dtype=jnp.int32)
            kv_batch_persist = jnp.tile(kv_packed, B)
            # Warmup
            _tout, _, _ = persistent_batched_decode_nlayer(
                w, config, first_tokens, PROMPT_LEN, kv_batch_persist, vocab_size, B, n_steps=5)
            _ = _tout.block_until_ready()

            persist_batch_times = []
            for _ in range(args.n_runs):
                t0 = time.perf_counter()
                tout, _, _ = persistent_batched_decode_nlayer(
                    w, config, first_tokens, PROMPT_LEN, kv_batch_persist, vocab_size, B, n_steps=GEN_LEN)
                _ = tout.tolist()
                persist_batch_times.append(time.perf_counter() - t0)
            pb_ms = np.median(persist_batch_times) * 1000
            pb_tok_s = B * GEN_LEN / pb_ms * 1000
            print(f"Total:            {pb_ms:.1f} ms ({GEN_LEN} steps)")
            print(f"Per step:         {pb_ms/GEN_LEN:.3f} ms")
            print(f"Throughput:       {pb_tok_s:.0f} tok/s total ({pb_tok_s/B:.0f} per seq)")
        except Exception as e:
            import traceback
            print(f"ERROR: {e}")
            traceback.print_exc()
        print()

    # Pipelined decode (no per-token int() sync — tokens stay on GPU)
    print(f"--- Decode: Pipelined ({GEN_LEN} tokens, grid={total_blocks}, no per-token sync) ---")
    try:
        # Warmup
        kv_tmp = kv_packed
        t = tok
        for i in range(5):
            t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
        _ = int(t)  # single sync

        pipe_times = []
        pipe_tokens_list = []
        for _ in range(args.n_runs):
            kv_tmp = kv_packed
            t = tok
            tok_devs = []
            t0 = time.perf_counter()
            for i in range(GEN_LEN):
                t, _, kv_tmp = multi_sm_decode_nlayer(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size)
                tok_devs.append(t)
            pipe_toks = [int(td) for td in tok_devs]  # batch sync at end
            pipe_times.append(time.perf_counter() - t0)
            pipe_tokens_list = pipe_toks
        pipe_ms = np.median(pipe_times) * 1000
        pipe_tok_s = GEN_LEN / pipe_ms * 1000
        pipe_ms_tok = pipe_ms / GEN_LEN
        print(f"Total:            {pipe_ms:.1f} ms")
        print(f"Per token:        {pipe_ms_tok:.3f} ms/tok")
        print(f"Throughput:       {pipe_tok_s:.0f} tok/s")
        print(f"Speedup vs sync:  {pipe_tok_s / tok_per_s:.2f}x")
        print(f"Text:             {decode_fn(pipe_tokens_list)[:100]}...")
        match_pipe = pipe_tokens_list == tokens
        print(f"Matches sync:     {'YES' if match_pipe else 'NO'}")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        pipe_tok_s = 0
    print()

    # Persistent decode (single kernel launch for all steps)
    print(f"--- Decode: Persistent ({GEN_LEN} tokens, grid={total_blocks}, single launch) ---")
    try:
        _tok_out, _, _ = persistent_decode_nlayer(
            w, config, tok, PROMPT_LEN, kv_packed, vocab_size, n_steps=5)
        _ = _tok_out.block_until_ready()

        persist_times = []
        persist_tokens = []
        for _ in range(args.n_runs):
            t0 = time.perf_counter()
            tok_out, _, _ = persistent_decode_nlayer(
                w, config, tok, PROMPT_LEN, kv_packed, vocab_size, n_steps=GEN_LEN)
            tok_list = tok_out.tolist()
            persist_times.append(time.perf_counter() - t0)
            persist_tokens = tok_list
        persist_ms = np.median(persist_times) * 1000
        persist_tok_s = GEN_LEN / persist_ms * 1000
        persist_ms_tok = persist_ms / GEN_LEN
        print(f"Total:            {persist_ms:.1f} ms")
        print(f"Per token:        {persist_ms_tok:.3f} ms/tok")
        print(f"Throughput:       {persist_tok_s:.0f} tok/s")
        print(f"Speedup vs sync:  {persist_tok_s / tok_per_s:.2f}x")
        print(f"Text:             {decode_fn(persist_tokens)[:100]}...")
        match_persist = persist_tokens == tokens
        print(f"Matches sync:     {'YES' if match_persist else 'NO'}")
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        persist_tok_s = 0
    print()

    # End-to-end
    total_ms = prefill_ms + decode_ms
    total_tokens = PROMPT_LEN + GEN_LEN
    print(f"--- End-to-End (Multi-SM) ---")
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
        "decode_tok_s_single_sm": round(tok_per_s_old, 0),
        "multi_sm_speedup": round(decode_ms_old / decode_ms, 2),
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
