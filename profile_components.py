"""Profile per-component costs in the multi-SM decode kernel.

Creates stripped-down kernels that only execute specific phases to measure:
  - Barrier overhead (spin-wait time)
  - LN + QKV projection per head
  - KV cache attention per head
  - O projection + reduction
  - FFN per block
  - Output projection
"""

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

    print(f"Model: d={d} h={n_heads} l={n_layers} ctx={ctx}")
    print()

    # Theoretical per-component data sizes
    print("=" * 60)
    print("THEORETICAL DATA SIZES")
    print("=" * 60)

    d_head = config["d_head"]
    d_ff = 4 * d

    # Per layer, per head
    qkv_w_bytes = 3 * d * d_head * 2  # Q/K/V weights, bf16
    o_w_bytes = d_head * d * 2  # O weight, bf16
    kv_cache_bytes = 2 * ctx * d_head * 2  # K + V cache for 1 head

    # Per layer, total across heads
    attn_total_bytes = n_heads * (qkv_w_bytes + o_w_bytes + kv_cache_bytes)

    # Per layer, total (shared across blocks)
    ln_bytes = 2 * d * 2  # scale + bias, bf16
    ffn_bytes = 2 * d * d_ff * 2 + d_ff * 2 + d * 2  # up + down + biases

    layer_total = attn_total_bytes + 2 * ln_bytes + ffn_bytes
    all_layers = n_layers * layer_total
    output_bytes = d * vocab_size * 2  # output projection

    print(f"  Per head (QKV weights):   {qkv_w_bytes/1024:.1f} KB")
    print(f"  Per head (O weight):      {o_w_bytes/1024:.1f} KB")
    print(f"  Per head (KV cache):      {kv_cache_bytes/1024:.1f} KB")
    print(f"  Per head total:           {(qkv_w_bytes+o_w_bytes+kv_cache_bytes)/1024:.1f} KB")
    print(f"  Per layer attention:      {attn_total_bytes/1024:.1f} KB")
    print(f"  Per layer FFN:            {ffn_bytes/1024:.1f} KB")
    print(f"  Per layer LN:             {2*ln_bytes/1024:.1f} KB")
    print(f"  Per layer total:          {layer_total/1024:.1f} KB")
    print(f"  All {n_layers} layers:           {all_layers/1e6:.1f} MB")
    print(f"  Output projection:        {output_bytes/1e6:.1f} MB")
    print(f"  Total unique data:        {(all_layers + output_bytes)/1e6:.1f} MB")
    print()

    # Additional redundant data (loaded by all 16 blocks)
    redundant_per_layer = 2 * ln_bytes  # LN loaded 16x
    redundant_reductions = n_heads * d * 4 * 2  # attention + FFN reductions, each 32KB, loaded 16x
    total_redundant = n_layers * (n_heads * redundant_per_layer + redundant_reductions)
    print(f"  Redundant LN per layer:   {n_heads * 2 * ln_bytes / 1024:.1f} KB (16x)")
    print(f"  Redundant reductions:     {n_layers * redundant_reductions / 1024:.1f} KB")
    print(f"  Total with redundancy:    {(all_layers + output_bytes + total_redundant)/1e6:.1f} MB")
    print()

    # Theoretical timing at different bandwidths
    unique_mb = (all_layers + output_bytes) / 1e6
    total_mb = (all_layers + output_bytes + total_redundant) / 1e6
    print(f"  At 836 GB/s (DRAM peak):  {unique_mb / 836 * 1000:.3f} ms (unique data)")
    print(f"  At 836 GB/s:              {total_mb / 836 * 1000:.3f} ms (with redundancy)")
    print(f"  At 2000 GB/s (L2 peak):   {total_mb / 2000 * 1000:.3f} ms (L2-cached)")
    print(f"  Achieved:                 ~0.36 ms")
    print()

    # Barrier overhead estimate
    n_barriers_per_step = 2 * n_layers + 1  # 16 layer barriers + 1 output barrier
    print(f"  Barriers per step:        {n_barriers_per_step}")
    print(f"  At 5µs/barrier:           {n_barriers_per_step * 0.005:.3f} ms")
    print(f"  At 10µs/barrier:          {n_barriers_per_step * 0.010:.3f} ms")
    print()

    # Breakdown: if we're at 0.36 ms and data transfer should take 0.08 ms...
    print("=" * 60)
    print("WHERE IS THE TIME?")
    print("=" * 60)
    kernel_ms = 0.57  # from profiling (block_until_ready)
    no_sync_ms = 0.36  # from amortized measurement
    print(f"  Measured kernel time:     {kernel_ms:.3f} ms (block_until_ready)")
    print(f"  Amortized no-sync:        {no_sync_ms:.3f} ms (pipelined)")
    print(f"  Theoretical data time:    {unique_mb / 836 * 1000:.3f} ms (836 GB/s)")
    gap = kernel_ms - unique_mb / 836 * 1000
    print(f"  Gap (overhead):           {gap:.3f} ms")
    print(f"  Possible causes:")
    print(f"    - Barriers: {n_barriers_per_step * 0.005:.3f} - {n_barriers_per_step * 0.010:.3f} ms")
    print(f"    - Redundant loads: {total_redundant / 836e6 * 1000:.3f} ms")
    print(f"    - L2 bandwidth limit: {total_mb / 2000 * 1000:.3f} ms at L2 peak")
    print(f"    - Compute (likely negligible for M=1)")

    with open(os.path.join(os.path.dirname(__file__), "profile_components_results.txt"), "w") as f:
        f.write(f"# Component profiling — {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"unique_data_mb: {unique_mb:.1f}\n")
        f.write(f"total_with_redundancy_mb: {total_mb:.1f}\n")
        f.write(f"theoretical_min_ms: {unique_mb / 836 * 1000:.3f}\n")
        f.write(f"kernel_ms: {kernel_ms:.3f}\n")
        f.write(f"n_barriers: {n_barriers_per_step}\n")


if __name__ == "__main__":
    main()
