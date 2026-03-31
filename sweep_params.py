"""Sweep num_warps and num_stages for multi-SM decode with kv_splits=2."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax_triton as jt

from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import _multi_sm_decode


def load_params():
    with open("weights.pkl", "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def decode_with_params(w, config, token_id, pos, kv_packed, vocab_size, kv_splits, num_warps, num_stages):
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_heads * max_seq * d_head
    total_blocks = n_heads * kv_splits
    ff_per_block = d_ff // total_blocks

    attn_ml_off = total_blocks * d_model
    barrier_off = attn_ml_off + total_blocks * 2
    argmax_off = barrier_off + 2 * n_layers + 1
    workspace_size = argmax_off + 2 * total_blocks

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
        grid=(total_blocks,),
        num_warps=num_warps, num_stages=num_stages,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_LAYERS=n_layers, MAX_SEQ=max_seq,
        KV_SPLITS=kv_splits, TOTAL_BLOCKS=total_blocks,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
        ATTN_ML_OFF=attn_ml_off,
        BARRIER_OFF=barrier_off,
        ARGMAX_OFF=argmax_off,
    )
    return next_token[0], logits_pad[:vocab_size], kv_out


def benchmark(w, config, tok, start_pos, kv_packed, vocab_size, kv_splits, num_warps, num_stages,
              n_tokens=128, n_runs=10):
    # Warmup
    kv = kv_packed
    t = tok
    for i in range(3):
        t, _, kv = decode_with_params(w, config, t, start_pos + i, kv, vocab_size,
                                       kv_splits, num_warps, num_stages)
        _ = int(t)

    times = []
    for _ in range(n_runs):
        kv = kv_packed
        t = tok
        t0 = time.perf_counter()
        for i in range(n_tokens):
            t, _, kv = decode_with_params(w, config, t, start_pos + i, kv, vocab_size,
                                           kv_splits, num_warps, num_stages)
            _ = int(t)
        times.append(time.perf_counter() - t0)
    return np.median(times) * 1000


def main():
    params, config = load_params()
    vocab_size = config["vocab_size"]
    prompt_len = 128
    gen_len = 128

    prompt = jnp.arange(prompt_len, dtype=jnp.int32) % vocab_size
    x = jnp.pad(prompt, (0, config["context_len"] - prompt_len)).astype(jnp.int32)
    logits, kc, vc = block_prefill(params, config, x, vocab_size)
    _ = logits.block_until_ready()

    kv_packed = pack_kv_caches(kc, vc)
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    tok = jnp.argmax(logits[prompt_len - 1])

    print("=== Parameter Sweep: kv_splits × num_warps × num_stages ===\n")
    results = []
    for kv_splits in [2]:
        for num_warps in [2, 4, 8]:
            for num_stages in [1, 2]:
                total_blocks = config["n_heads"] * kv_splits
                try:
                    ms = benchmark(w, config, tok, prompt_len, kv_packed, vocab_size,
                                  kv_splits, num_warps, num_stages, n_tokens=gen_len)
                    tok_s = gen_len / ms * 1000
                    ms_tok = ms / gen_len
                    results.append((kv_splits, num_warps, num_stages, tok_s, ms_tok))
                    print(f"kv={kv_splits} warps={num_warps} stages={num_stages} grid={total_blocks}: "
                          f"{tok_s:.0f} tok/s  ({ms_tok:.3f} ms/tok)")
                except Exception as e:
                    print(f"kv={kv_splits} warps={num_warps} stages={num_stages}: FAILED — {e}")

    print("\n=== Best Configuration ===")
    best = max(results, key=lambda x: x[3])
    print(f"kv_splits={best[0]}, num_warps={best[1]}, num_stages={best[2]}: {best[3]:.0f} tok/s")

    with open("sweep_results.txt", "w") as f:
        f.write("kv_splits,num_warps,num_stages,tok_s,ms_per_tok\n")
        for r in results:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]:.0f},{r[4]:.3f}\n")
    print("\nSaved to sweep_results.txt")


if __name__ == "__main__":
    main()
