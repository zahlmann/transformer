"""Speculative decoding: draft model proposes K tokens, target model verifies in parallel.

Uses the small (d=128, 1L) model as draft and the large (d=256, 4L) model as target.
Both share the same BPE vocabulary (vocab=4096, TinyStories).

Algorithm (greedy):
  1. Draft model generates K tokens autoregressively (fast, ~3000+ tok/s)
  2. Target model verifies all K tokens in one parallel forward pass
  3. Find first disagreement position; accept all tokens before it
  4. Use target's token at the disagreement position
  5. Repeat from step 1

The verification forward pass reuses block_prefill kernels with a modified
attention mask that attends to the existing KV cache prefix.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp

from data import prepare_data, load_bpe_vocab
from model import transformer_forward, count_params
from kernels.block_prefill import block_prefill
from kernels.fused_decode_nlayer import (
    fused_decode_nlayer, prepare_decode_weights_nlayer, pack_kv_caches, unpack_kv_caches)


def load_model(path):
    with open(path, "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def generate_standard(w, config, prompt, n_tokens, vocab_size):
    """Standard autoregressive decode with fused N-layer kernel."""
    x = jnp.pad(prompt, (0, config["context_len"] - len(prompt))).astype(jnp.int32)
    logits, k_caches, v_caches = block_prefill(w, config, x, vocab_size)
    kv_packed = pack_kv_caches(k_caches, v_caches)

    wd = prepare_decode_weights_nlayer(w, config, vocab_size)
    tokens = []
    tok = jnp.argmax(logits[len(prompt) - 1])
    tokens.append(int(tok))

    for i in range(n_tokens - 1):
        logits, kv_packed = fused_decode_nlayer(
            wd, config, tok, len(prompt) + i, kv_packed, vocab_size)
        tok = jnp.argmax(logits)
        tokens.append(int(tok))
    return tokens


def generate_speculative(draft_params, draft_config, target_params, target_config,
                         prompt, n_tokens, vocab_size, K=4):
    """Speculative decoding: draft proposes K tokens, target verifies in parallel.

    Uses greedy decoding (argmax) for both draft and target. Acceptance is binary:
    if draft token matches target's argmax, accept; otherwise reject and use target's.
    """
    ctx_len_target = target_config["context_len"]
    ctx_len_draft = draft_config["context_len"]
    n_layers_target = target_config["n_layers"]
    n_heads_target = target_config["n_heads"]
    d_head_target = target_config["d_head"]

    # Prefill both models with the prompt
    x_target = jnp.pad(prompt, (0, ctx_len_target - len(prompt))).astype(jnp.int32)
    x_draft = jnp.pad(prompt, (0, ctx_len_draft - len(prompt))).astype(jnp.int32)

    t_logits, t_k_caches, t_v_caches = block_prefill(target_params, target_config, x_target, vocab_size)
    d_logits, d_k_caches, d_v_caches = block_prefill(draft_params, draft_config, x_draft, vocab_size)

    t_kv = pack_kv_caches(t_k_caches, t_v_caches)
    d_kv = pack_kv_caches(d_k_caches, d_v_caches)

    t_w = prepare_decode_weights_nlayer(target_params, target_config, vocab_size)
    d_w = prepare_decode_weights_nlayer(draft_params, draft_config, vocab_size)

    # First token from target model
    tok = int(jnp.argmax(t_logits[len(prompt) - 1]))
    # Also advance draft to same token
    d_tok = int(jnp.argmax(d_logits[len(prompt) - 1]))

    tokens = [tok]
    pos = len(prompt)  # next position to fill

    total_draft = 0
    total_accepted = 0

    while len(tokens) < n_tokens:
        remaining = n_tokens - len(tokens)
        k = min(K, remaining)

        # Step 1: Draft generates K tokens
        draft_tokens = []
        draft_tok = tok
        draft_kv_tmp = d_kv
        for j in range(k):
            d_logits_j, draft_kv_tmp = fused_decode_nlayer(
                d_w, draft_config, draft_tok, pos + j, draft_kv_tmp, vocab_size)
            draft_tok = int(jnp.argmax(d_logits_j))
            draft_tokens.append(draft_tok)

        total_draft += k

        # Step 2: Target verifies K tokens sequentially
        # (We'll optimize this with a parallel kernel later — task #3)
        target_logits_list = []
        verify_kv = t_kv
        verify_tok = tok
        for j in range(k):
            t_logits_j, verify_kv = fused_decode_nlayer(
                t_w, target_config, verify_tok, pos + j, verify_kv, vocab_size)
            target_logits_list.append(t_logits_j)
            verify_tok = draft_tokens[j]

        # Step 3: Find first disagreement
        n_accepted = 0
        for j in range(k):
            target_tok = int(jnp.argmax(target_logits_list[j]))
            if target_tok == draft_tokens[j]:
                n_accepted += 1
            else:
                # Reject: use target's token at this position
                tokens.append(target_tok)
                pos += n_accepted + 1
                total_accepted += n_accepted

                # Roll back draft KV to accepted prefix
                # Re-advance draft through accepted tokens + target's correction
                d_kv_rollback = d_kv
                rollback_tok = tok
                for jj in range(n_accepted + 1):
                    actual_tok = draft_tokens[jj] if jj < n_accepted else target_tok
                    _, d_kv_rollback = fused_decode_nlayer(
                        d_w, draft_config, rollback_tok, pos - n_accepted - 1 + jj,
                        d_kv_rollback, vocab_size)
                    rollback_tok = actual_tok

                # Update state
                t_kv = verify_kv
                d_kv = d_kv_rollback
                # Add accepted draft tokens before the rejection
                for jj in range(n_accepted):
                    tokens.insert(-1, draft_tokens[jj])
                tok = target_tok
                break
        else:
            # All K tokens accepted
            total_accepted += k
            for dt in draft_tokens:
                tokens.append(dt)
            pos += k

            # Get one bonus token from target at position pos-1
            # Target already processed all K draft tokens, get its next prediction
            t_logits_bonus, verify_kv = fused_decode_nlayer(
                t_w, target_config, draft_tokens[-1], pos, verify_kv, vocab_size)
            bonus_tok = int(jnp.argmax(t_logits_bonus))
            tokens.append(bonus_tok)
            pos += 1

            # Advance draft through the bonus token
            _, draft_kv_tmp = fused_decode_nlayer(
                d_w, draft_config, draft_tokens[-1], pos - 1, draft_kv_tmp, vocab_size)

            t_kv = verify_kv
            d_kv = draft_kv_tmp
            tok = bonus_tok

    tokens = tokens[:n_tokens]
    acceptance_rate = total_accepted / max(total_draft, 1)
    return tokens, acceptance_rate, total_accepted, total_draft


def bench(fn, n_warmup, n_iters):
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, tuple):
            _ = int(result[0][-1])
        else:
            _ = int(result[-1])
        times.append(time.perf_counter() - t0)
    return times, result


def main():
    # Load both models
    draft_params, draft_config = load_model("draft_weights.pkl")
    target_params, target_config = load_model("target_weights.pkl")
    vocab_size = target_config["vocab_size"]

    bpe_vocab = load_bpe_vocab()
    decode = bpe_vocab["decode_fn"]
    dataset = "tinystories"
    data = prepare_data(tokenizer="trained_bpe", bpe_vocab_size=vocab_size, dataset=dataset)

    PROMPT_LEN = 128
    GEN_LEN = 128
    WARMUP = 3
    BENCH_ITERS = 10

    prompt = jnp.array(data["train_x"][0][:PROMPT_LEN], dtype=jnp.int32)
    print(f"Draft:  d={draft_config['d_model']} h={draft_config['n_heads']} l={draft_config['n_layers']} params={count_params(draft_params):,}")
    print(f"Target: d={target_config['d_model']} h={target_config['n_heads']} l={target_config['n_layers']} params={count_params(target_params):,}")
    print(f"Prompt: {decode(prompt)[:200]}...\n")

    # Benchmark standard target decode
    print("=== Standard Target Decode ===")
    std_times, std_tokens = bench(
        lambda: generate_standard(target_params, target_config, prompt, GEN_LEN, vocab_size),
        WARMUP, BENCH_ITERS)
    std_ms = np.mean(std_times) * 1000
    std_tps = GEN_LEN / np.mean(std_times)
    print(f"  {std_tps:.0f} tok/s  ({std_ms:.1f}ms)  text: {decode(std_tokens)[:200]}...")

    # Benchmark speculative decode for different K values
    for K in [2, 3, 4, 6, 8]:
        print(f"\n=== Speculative Decode K={K} ===")
        spec_times, spec_result = bench(
            lambda K=K: generate_speculative(
                draft_params, draft_config, target_params, target_config,
                prompt, GEN_LEN, vocab_size, K=K),
            WARMUP, BENCH_ITERS)
        spec_tokens, acc_rate, n_acc, n_draft = spec_result
        spec_ms = np.mean(spec_times) * 1000
        spec_tps = GEN_LEN / np.mean(spec_times)
        speedup = spec_tps / std_tps
        print(f"  {spec_tps:.0f} tok/s  ({spec_ms:.1f}ms)  speedup={speedup:.2f}x")
        print(f"  acceptance={acc_rate:.1%}  accepted={n_acc}/{n_draft}")
        print(f"  text: {decode(spec_tokens)[:200]}...")

    # Save results
    with open("speculative_results.txt", "w") as f:
        f.write(f"Draft:  d={draft_config['d_model']} l={draft_config['n_layers']} params={count_params(draft_params):,}\n")
        f.write(f"Target: d={target_config['d_model']} l={target_config['n_layers']} params={count_params(target_params):,}\n")
        f.write(f"Standard: {std_tps:.0f} tok/s ({std_ms:.1f}ms)\n")


if __name__ == "__main__":
    main()
