"""Streaming text generation using Triton decode kernels.

Usage:
    # Interactive CLI (greedy)
    uv run generate.py --prompt "Once upon a time"

    # With sampling
    uv run generate.py --prompt "Once upon a time" --temp 0.8 --top-p 0.95

    # With repetition penalty
    uv run generate.py --prompt "Once upon a time" --temp 0.8 --top-p 0.95 --rep-penalty 1.2

    # Non-streaming (faster, pipelined)
    uv run generate.py --prompt "Once upon a time" --no-stream
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer
from model import prefill_with_kv


def sample_token(logits, temperature=1.0, top_p=1.0, rep_penalty=1.0,
                 generated_ids=None, rng_key=None):
    """Sample a token from logits with temperature, top-p, and repetition penalty.

    Args:
        logits: (vocab_size,) raw logits
        temperature: >0, lower = more deterministic, 1.0 = neutral
        top_p: nucleus sampling threshold (1.0 = disabled)
        rep_penalty: >1.0 penalizes already-generated tokens
        generated_ids: list of previously generated token IDs
        rng_key: JAX PRNG key for sampling
    Returns:
        token_id (int)
    """
    logits = np.array(logits, dtype=np.float32)

    # repetition penalty: reduce logits of already-seen tokens
    if rep_penalty != 1.0 and generated_ids:
        seen = list(set(generated_ids))
        for tid in seen:
            if logits[tid] > 0:
                logits[tid] /= rep_penalty
            else:
                logits[tid] *= rep_penalty

    # greedy fast path
    if temperature == 0.0 or (temperature == 1.0 and top_p == 1.0 and rep_penalty == 1.0
                               and not generated_ids):
        return int(np.argmax(logits))

    # temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # softmax
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()

    # top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_idx = np.argsort(-probs)
        sorted_probs = probs[sorted_idx]
        cumsum = np.cumsum(sorted_probs)
        # keep tokens up to cumulative probability >= top_p
        cutoff = np.searchsorted(cumsum, top_p) + 1
        keep_idx = sorted_idx[:cutoff]
        filtered_probs = probs[keep_idx]
        filtered_probs /= filtered_probs.sum()
        return int(np.random.choice(keep_idx, p=filtered_probs))

    return int(np.random.choice(len(probs), p=probs))


def _prefill(params, config, prompt_ids, vocab_size):
    """Run prefill and return (weights, first_logits, start_pos, kv_packed).

    Includes a warmup decode step to trigger Triton JIT compilation.
    """
    ctx_len = config["context_len"]
    prompt_len = len(prompt_ids)
    x = jnp.pad(prompt_ids, (0, ctx_len - prompt_len)).astype(jnp.int32)

    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()

    w = prepare_decode_weights_nlayer(params, config, vocab_size, kv_splits=1)
    kv_packed = pack_kv_caches(k_caches, v_caches)
    first_logits = logits[prompt_len - 1]

    # Warmup decode step to trigger Triton JIT compilation
    warmup_tok = jnp.argmax(first_logits)
    _tok, _, _kv = multi_sm_decode_nlayer(
        w, config, warmup_tok, prompt_len, kv_packed, vocab_size, kv_splits=1)
    _ = int(_tok)

    return w, first_logits, prompt_len, kv_packed


def stream_tokens(params, config, prompt_ids, max_tokens=128,
                  temperature=0.0, top_p=1.0, rep_penalty=1.0, seed=None):
    """Yield token IDs one at a time with optional sampling."""
    if seed is not None:
        np.random.seed(seed)
    vocab_size = config["vocab_size"]
    w, first_logits, start_pos, kv_packed = _prefill(
        params, config, prompt_ids, vocab_size)

    generated = []

    # First token from prefill logits
    if temperature == 0.0 and rep_penalty == 1.0:
        first_id = int(jnp.argmax(first_logits))
    else:
        first_id = sample_token(first_logits, temperature, top_p,
                                rep_penalty, generated)
    generated.append(first_id)
    yield first_id

    tok = jnp.int32(first_id)
    for i in range(max_tokens - 1):
        tok_out, logits, kv_packed = multi_sm_decode_nlayer(
            w, config, tok, start_pos + i, kv_packed, vocab_size, kv_splits=1)

        if temperature == 0.0 and rep_penalty == 1.0:
            token_id = int(tok_out)
        else:
            token_id = sample_token(logits, temperature, top_p,
                                    rep_penalty, generated)

        generated.append(token_id)
        yield token_id
        tok = jnp.int32(token_id)


def generate_tokens(params, config, prompt_ids, n_tokens=128,
                    temperature=0.0, top_p=1.0, rep_penalty=1.0, seed=None):
    """Generate all tokens at once."""
    return list(stream_tokens(params, config, prompt_ids, n_tokens,
                              temperature, top_p, rep_penalty, seed))


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
    parser.add_argument("--temp", type=float, default=0.0,
                        help="Sampling temperature (0 = greedy, 0.7-1.0 typical)")
    parser.add_argument("--top-p", type=float, default=1.0,
                        help="Nucleus sampling threshold (0.9-0.95 typical)")
    parser.add_argument("--rep-penalty", type=float, default=1.0,
                        help="Repetition penalty (1.0 = off, 1.1-1.3 typical)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible sampling")
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
    if args.temp > 0:
        print(f"Sampling: temp={args.temp} top_p={args.top_p} "
              f"rep_penalty={args.rep_penalty}", file=sys.stderr)
    else:
        print(f"Decoding: greedy (argmax)", file=sys.stderr)
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
        tokens = generate_tokens(params, config, prompt_ids, max_gen,
                                 args.temp, args.top_p, args.rep_penalty, args.seed)
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
        for token_id in stream_tokens(params, config, prompt_ids, max_gen,
                                      args.temp, args.top_p, args.rep_penalty, args.seed):
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
