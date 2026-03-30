"""Test different num_warps settings for the multi-SM kernel."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax_triton as jt

from model import count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import _multi_sm_decode


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def decode_with_warps(w, config, token_id, pos, kv_packed, vocab_size, num_warps):
    """Call the multi-SM kernel with a specific num_warps."""
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_heads * max_seq * d_head
    ff_per_block = d_ff // n_heads
    barrier_off = n_heads * d_model
    argmax_off = barrier_off + 2 * n_layers + 1
    workspace_size = argmax_off + 2 * n_heads

    workspace = jnp.zeros((workspace_size,), dtype=jnp.float32)

    logits_pad, kv_out, next_token = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
        jnp.int32(token_id), jnp.int32(pos),
        kv_packed,
        workspace,
        kernel=_multi_sm_decode,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ],
        grid=(n_heads,),
        num_warps=num_warps, num_stages=1,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_LAYERS=n_layers, MAX_SEQ=max_seq,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
        BARRIER_OFF=barrier_off,
        ARGMAX_OFF=argmax_off,
    )
    return next_token[0], logits_pad[:vocab_size], kv_out


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    PROMPT_LEN = 128
    GEN_LEN = 64

    prompt = jnp.arange(PROMPT_LEN, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - PROMPT_LEN)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()
    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[PROMPT_LEN - 1])

    print(f"Model: d={config['d_model']} h={config['n_heads']} l={config['n_layers']}")
    print()

    for num_warps in [2, 4, 8, 16]:
        print(f"--- num_warps={num_warps} ---")
        try:
            # Warmup
            kv_tmp = kv_packed
            t = tok
            for i in range(5):
                t, _, kv_tmp = decode_with_warps(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size, num_warps)
                _ = int(t)

            # Benchmark (no sync)
            kv_tmp = kv_packed
            t = tok
            t0 = time.perf_counter()
            for i in range(GEN_LEN):
                t, _, kv_tmp = decode_with_warps(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size, num_warps)
            _ = int(t)
            no_sync_ms = (time.perf_counter() - t0) * 1000

            # Benchmark (with sync)
            kv_tmp = kv_packed
            t = tok
            t0 = time.perf_counter()
            for i in range(GEN_LEN):
                t, _, kv_tmp = decode_with_warps(w, config, t, PROMPT_LEN + i, kv_tmp, vocab_size, num_warps)
                _ = int(t)
            sync_ms = (time.perf_counter() - t0) * 1000

            print(f"  With sync:    {sync_ms:.1f} ms  ({GEN_LEN/sync_ms*1000:.0f} tok/s)")
            print(f"  Without sync: {no_sync_ms:.1f} ms  ({GEN_LEN/no_sync_ms*1000:.0f} tok/s)")
        except Exception as e:
            print(f"  FAILED: {e}")
        print()


if __name__ == "__main__":
    main()
