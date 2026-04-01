"""Streaming text generation using Triton decode kernels.

Usage:
    # Stream tokens one at a time
    for token_id in stream_tokens(params, config, prompt, max_tokens=128):
        print(decode_fn(token_id), end='', flush=True)

    # Generate all tokens at once (fastest — pipelined, no per-token sync)
    tokens = generate_tokens(params, config, prompt, n_tokens=128)

    # Interactive CLI
    uv run generate.py --prompt "Once upon a time"
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import jax
import jax.numpy as jnp

from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer
from model import prefill_with_kv


def _prefill(params, config, prompt_ids, vocab_size):
    """Run prefill and return (weights, first_token, start_pos, kv_packed).

    Includes a warmup decode step to trigger Triton JIT compilation.
    """
    ctx_len = config["context_len"]
    prompt_len = len(prompt_ids)
    x = jnp.pad(prompt_ids, (0, ctx_len - prompt_len)).astype(jnp.int32)

    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()

    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    kv_packed = pack_kv_caches(k_caches, v_caches)
    first_token = jnp.argmax(logits[prompt_len - 1])

    # Warmup decode step to trigger Triton JIT compilation
    _tok, _, _kv = multi_sm_decode_nlayer(
        w, config, first_token, prompt_len, kv_packed, vocab_size)
    _ = int(_tok)

    return w, first_token, prompt_len, kv_packed


def stream_tokens(params, config, prompt_ids, max_tokens=128, vocab_size=None):
    """Yield token IDs one at a time as they're generated.

    Uses pipelined multi-SM decode (one kernel call per token).
    Each token is synced to host immediately for streaming output.

    Args:
        params: model parameters
        config: model config dict
        prompt_ids: jnp.array of prompt token IDs
        max_tokens: maximum tokens to generate
        vocab_size: vocabulary size (inferred from config if None)

    Yields:
        int: token IDs, one at a time
    """
    if vocab_size is None:
        vocab_size = config["vocab_size"]

    w, tok, start_pos, kv_packed = _prefill(params, config, prompt_ids, vocab_size)

    # First token (from prefill argmax)
    first_id = int(tok)
    yield first_id

    # Decode remaining tokens
    for i in range(max_tokens - 1):
        tok, _, kv_packed = multi_sm_decode_nlayer(
            w, config, tok, start_pos + i, kv_packed, vocab_size)
        yield int(tok)


def generate_tokens(params, config, prompt_ids, n_tokens=128, vocab_size=None):
    """Generate all tokens at once (pipelined, minimal per-token sync).

    Faster than stream_tokens because tokens stay on device between steps.
    All tokens synced to host at the end.

    Args:
        params: model parameters
        config: model config dict
        prompt_ids: jnp.array of prompt token IDs
        n_tokens: number of tokens to generate
        vocab_size: vocabulary size (inferred from config if None)

    Returns:
        list[int]: generated token IDs
    """
    if vocab_size is None:
        vocab_size = config["vocab_size"]

    w, tok, start_pos, kv_packed = _prefill(params, config, prompt_ids, vocab_size)

    # Collect device-side token tensors (no per-step sync)
    tok_devs = [tok]
    for i in range(n_tokens - 1):
        tok, _, kv_packed = multi_sm_decode_nlayer(
            w, config, tok, start_pos + i, kv_packed, vocab_size)
        tok_devs.append(tok)

    # Single batch sync at end
    return [int(t) for t in tok_devs]


def main():
    import argparse
    import pickle
    import sys
    import time

    from data import load_bpe_vocab

    parser = argparse.ArgumentParser(description="Generate text with Triton kernels")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Text prompt to continue")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum tokens to generate")
    parser.add_argument("--weights", type=str, default="weights.pkl",
                        help="Path to weights file")
    parser.add_argument("--no-stream", action="store_true",
                        help="Generate all tokens at once (faster, no streaming)")
    args = parser.parse_args()

    # Load model
    with open(os.path.join(os.path.dirname(__file__), args.weights), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    vocab_size = config["vocab_size"]

    d = config["d_model"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    ctx_len = config["context_len"]

    # Load tokenizer
    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(bpe_vocab["tokenizer_path"])
    encode_fn = lambda text: tok.encode(text).ids

    print(f"Model: d={d} h={n_heads} l={n_layers} ctx={ctx_len}", file=sys.stderr)
    print(f"Params: {sum(p.size for p in params.values()):,}", file=sys.stderr)
    print(file=sys.stderr)

    # Encode prompt
    prompt_ids = encode_fn(args.prompt)
    if len(prompt_ids) >= ctx_len:
        prompt_ids = prompt_ids[:ctx_len - 1]
    prompt_ids = jnp.array(prompt_ids, dtype=jnp.int32)
    max_gen = min(args.max_tokens, ctx_len - len(prompt_ids))

    print(f"Prompt ({len(prompt_ids)} tokens): {args.prompt}", file=sys.stderr)
    print(f"Generating {max_gen} tokens...", file=sys.stderr)
    print(file=sys.stderr)

    # Print prompt, then stream generated tokens
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    if args.no_stream:
        t0 = time.perf_counter()
        tokens = generate_tokens(params, config, prompt_ids, max_gen, vocab_size)
        elapsed = time.perf_counter() - t0
        text = decode_fn(tokens)
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
        tok_per_s = max_gen / elapsed
        print(f"\n[{max_gen} tokens in {elapsed*1000:.0f}ms = {tok_per_s:.0f} tok/s"
              f" (includes JIT compilation on first run)]", file=sys.stderr)
    else:
        t0 = time.perf_counter()
        all_tokens = []
        t_first = None
        printed_chars = 0
        for token_id in stream_tokens(params, config, prompt_ids, max_gen, vocab_size):
            if t_first is None:
                t_first = time.perf_counter()
            all_tokens.append(token_id)
            # Decode just the new token for display (fast path)
            new_text = decode_fn([token_id])
            if new_text:
                sys.stdout.write(new_text)
                sys.stdout.flush()
                printed_chars += len(new_text)
        elapsed = time.perf_counter() - t0
        sys.stdout.write("\n")
        sys.stdout.flush()
        ttft = (t_first - t0) * 1000 if t_first else 0
        decode_time = elapsed - (t_first - t0) if t_first else elapsed
        decode_tok_s = (max_gen - 1) / decode_time if decode_time > 0 else 0
        print(f"\n[{max_gen} tokens, TTFT={ttft:.0f}ms (includes compilation),"
              f" decode={decode_tok_s:.0f} tok/s]", file=sys.stderr)


if __name__ == "__main__":
    main()
