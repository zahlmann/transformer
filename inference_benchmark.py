"""Inference benchmark: fused Triton kernels vs JAX/XLA baseline."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import prepare_data, load_bpe_vocab
from model import transformer_forward, count_params

PROMPT_LEN = 128
GEN_LEN = 128
WARMUP = 5
BENCH_ITERS = 20


def load_params():
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    tokenizer = saved.get("tokenizer", "char")
    return params, config, tokenizer


def generate_triton(params, config, prompt, n_tokens, vocab_size):
    """Generate tokens using the appropriate Triton kernels based on model size."""
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    d_head = config["d_head"]
    ctx_len = config["context_len"]

    if d_model <= 64 and n_layers == 1:
        # Small model: single fused kernel
        from kernels.fused_prefill import fused_prefill
        from kernels.fused_decode import fused_decode, prepare_decode_weights_small

        w = prepare_decode_weights_small(params, vocab_size)
        x = jnp.pad(prompt, (0, ctx_len - len(prompt))).astype(jnp.int32)
        logits, kc, vc = fused_prefill(params, x, vocab_size=vocab_size)
        tokens = []
        tok = jnp.argmax(logits[len(prompt) - 1])
        tokens.append(int(tok))
        for i in range(n_tokens - 1):
            logits, kc, vc = fused_decode(w, tok, len(prompt) + i, kc, vc, vocab_size=vocab_size)
            tok = jnp.argmax(logits)
            tokens.append(int(tok))
        return tokens
    else:
        # Larger model: multi-block prefill + fused decode
        from kernels.block_prefill import block_prefill

        x = jnp.pad(prompt, (0, ctx_len - len(prompt))).astype(jnp.int32)
        logits, k_caches, v_caches = block_prefill(params, config, x, vocab_size)
        tokens = []
        tok = jnp.argmax(logits[len(prompt) - 1])
        tokens.append(int(tok))

        if n_layers == 2:
            from kernels.fused_decode_2layer import fused_decode_2layer, prepare_decode_weights
            w = prepare_decode_weights(params, config, vocab_size)
            for i in range(n_tokens - 1):
                logits, k_caches, v_caches = fused_decode_2layer(
                    w, config, tok, len(prompt) + i,
                    k_caches, v_caches, vocab_size)
                tok = jnp.argmax(logits)
                tokens.append(int(tok))
        else:
            from kernels.block_decode import block_decode, prepare_decode_weights_block
            w = prepare_decode_weights_block(params, config, vocab_size)
            for i in range(n_tokens - 1):
                logits, k_caches, v_caches = block_decode(
                    w, config, tok, len(prompt) + i,
                    k_caches, v_caches, vocab_size)
                tok = jnp.argmax(logits)
                tokens.append(int(tok))
        return tokens


def generate_jax(fwd, prompt, n_tokens):
    seq = list(prompt)
    for _ in range(n_tokens):
        logits = fwd(jnp.array(seq, dtype=jnp.int32))
        seq.append(int(jnp.argmax(logits[-1])))
    return seq[len(prompt):]


def bench(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        _ = int(result[-1])  # force sync
        times.append(time.perf_counter() - t0)
    return times, result


def main():
    params, config, tokenizer = load_params()
    vocab_size = config["vocab_size"]

    if tokenizer == "trained_bpe":
        bpe_vocab = load_bpe_vocab()
        decode = bpe_vocab["decode_fn"]
        # Detect dataset from tokenizer path
        tok_path = bpe_vocab.get("tokenizer_path", "")
        dataset = "tinystories" if "tinystories" in tok_path else "shakespeare"
        data = prepare_data(tokenizer="trained_bpe", bpe_vocab_size=vocab_size, dataset=dataset)
    elif tokenizer == "bpe":
        bpe_vocab = load_bpe_vocab()
        compact_to_bytes = bpe_vocab["compact_to_bytes"]
        decode = lambda ids: b"".join(compact_to_bytes.get(int(i), b"?") for i in ids).decode("utf-8", errors="replace")
        data = prepare_data(tokenizer="bpe", bpe_vocab_size=vocab_size)
    else:
        data = prepare_data(tokenizer="char")
        chars = data["chars"]
        decode = lambda ids: ''.join(chars[i] for i in ids)

    prompt = jnp.array(data["train_x"][0][:PROMPT_LEN], dtype=jnp.int32)
    print(f"Model: d_model={config['d_model']}, n_heads={config['n_heads']}, n_layers={config['n_layers']}")
    print(f"Tokenizer: {tokenizer}, vocab_size: {vocab_size}, params: {count_params(params):,}")
    print(f"Prompt: {decode(prompt)}\n")

    # Triton
    tri_times, tri_tokens = bench(
        lambda: generate_triton(params, config, prompt, GEN_LEN, vocab_size), WARMUP, BENCH_ITERS)
    tri_ms = np.mean(tri_times) * 1000
    tri_tps = GEN_LEN / np.mean(tri_times)
    print(f"Triton:  {tri_tps:>6.0f} tok/s  ({tri_ms:.1f}ms)  text: {decode(tri_tokens)}")

    # JAX baseline
    fwd = jax.jit(lambda x: transformer_forward(params, config, x))
    jax_times, jax_tokens = bench(
        lambda: generate_jax(fwd, prompt, GEN_LEN), WARMUP, BENCH_ITERS)
    jax_ms = np.mean(jax_times) * 1000
    jax_tps = GEN_LEN / np.mean(jax_times)
    print(f"JAX:     {jax_tps:>6.0f} tok/s  ({jax_ms:.1f}ms)  text: {decode(jax_tokens)}")

    speedup = jax_ms / tri_ms
    print(f"\nSpeedup: {speedup:.2f}x")

    # Save
    with open(os.path.join(os.path.dirname(__file__), "inference_results.txt"), "w") as f:
        f.write(f"Model: d={config['d_model']} h={config['n_heads']} l={config['n_layers']}\n")
        f.write(f"Tokenizer: {tokenizer}, vocab_size: {vocab_size}\n")
        f.write(f"Triton: {tri_tps:.0f} tok/s ({tri_ms:.1f}ms)\n")
        f.write(f"JAX:    {jax_tps:.0f} tok/s ({jax_ms:.1f}ms)\n")
        f.write(f"Speedup: {speedup:.2f}x\n")


if __name__ == "__main__":
    main()
