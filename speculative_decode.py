"""Speculative decoding revisit: d=512 draft + d=768 target.

Measures acceptance rates and throughput for speculative decoding using the
d=512 model as draft and d=768 as target. Both use the multi-SM decode kernel.

Key question: with close perplexity (2.91 vs 2.60), is acceptance high enough
to overcome the overhead of sequential verification?

Math for sequential verification (K=4):
  Draft:  4 × 0.55ms = 2.2ms  (d=512 sync'd)
  Verify: 4 × 1.0ms  = 4.0ms  (d=768 sync'd)
  Total:  6.2ms for ~4 tokens → 645 tok/s at 60% acceptance
  Standard d=768: ~1000 tok/s → NOT profitable with sequential verify

So this primarily benchmarks acceptance rates to determine if building a
parallel verification kernel would be worthwhile.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from model import prefill_with_kv
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_model(path):
    with open(os.path.join(os.path.dirname(__file__), path), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    return params, config


def prefill(params, config, prompt_ids, vocab_size):
    ctx_len = config["context_len"]
    prompt_len = len(prompt_ids)
    x = jnp.pad(jnp.array(prompt_ids, dtype=jnp.int32),
                 (0, ctx_len - prompt_len)).astype(jnp.int32)
    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    kv = pack_kv_caches(k_caches, v_caches)
    first_token = int(jnp.argmax(logits[prompt_len - 1]))
    return w, first_token, prompt_len, kv


def generate_standard(w, config, first_token, prompt_len, kv, vocab_size, n_tokens):
    """Standard autoregressive decode (baseline)."""
    tokens = [first_token]
    tok = first_token
    for i in range(n_tokens - 1):
        tok, _, kv = multi_sm_decode_nlayer(
            w, config, tok, prompt_len + i, kv, vocab_size)
        tok = int(tok)
        tokens.append(tok)
    return tokens, kv


def generate_speculative(draft_w, draft_config, target_w, target_config,
                         draft_first, target_first, prompt_len,
                         draft_kv, target_kv, vocab_size, n_tokens, K=4):
    """Speculative decoding with sequential target verification."""
    tok = target_first
    tokens = [tok]
    pos = prompt_len

    d_kv = draft_kv
    t_kv = target_kv

    total_draft = 0
    total_accepted = 0
    cycle_count = 0

    while len(tokens) < n_tokens:
        k = min(K, n_tokens - len(tokens))
        cycle_count += 1

        # Draft K tokens
        draft_tokens = []
        draft_tok = tok
        draft_kv_tmp = d_kv
        for j in range(k):
            draft_tok, _, draft_kv_tmp = multi_sm_decode_nlayer(
                draft_w, draft_config, draft_tok, pos + j, draft_kv_tmp, vocab_size)
            draft_tok = int(draft_tok)
            draft_tokens.append(draft_tok)
        total_draft += k

        # Verify K tokens sequentially through target
        target_toks = []
        verify_kv = t_kv
        verify_tok = tok
        for j in range(k):
            verify_tok_new, _, verify_kv = multi_sm_decode_nlayer(
                target_w, target_config, verify_tok, pos + j, verify_kv, vocab_size)
            target_toks.append(int(verify_tok_new))
            verify_tok = draft_tokens[j]  # feed draft token as next input

        # Find first disagreement
        n_acc = 0
        for j in range(k):
            if target_toks[j] == draft_tokens[j]:
                n_acc += 1
            else:
                break
        total_accepted += n_acc

        if n_acc == k:
            # All accepted — add draft tokens + bonus
            tokens.extend(draft_tokens)
            pos += k
            bonus_tok, _, verify_kv = multi_sm_decode_nlayer(
                target_w, target_config, draft_tokens[-1], pos, verify_kv, vocab_size)
            bonus = int(bonus_tok)
            tokens.append(bonus)
            pos += 1
            # Advance draft past the bonus
            _, _, draft_kv_tmp = multi_sm_decode_nlayer(
                draft_w, draft_config, draft_tokens[-1], pos - 1, draft_kv_tmp, vocab_size)
            t_kv = verify_kv
            d_kv = draft_kv_tmp
            tok = bonus
        else:
            # Partial accept
            accepted = draft_tokens[:n_acc]
            correction = target_toks[n_acc]
            tokens.extend(accepted)
            tokens.append(correction)
            pos += n_acc + 1

            # Re-advance draft through accepted + correction
            d_kv_rb = d_kv
            rb_tok = tok
            for jj in range(n_acc + 1):
                actual = draft_tokens[jj] if jj < n_acc else correction
                _, _, d_kv_rb = multi_sm_decode_nlayer(
                    draft_w, draft_config, rb_tok, pos - n_acc - 1 + jj,
                    d_kv_rb, vocab_size)
                rb_tok = actual
            t_kv = verify_kv
            d_kv = d_kv_rb
            tok = correction

    acc_rate = total_accepted / max(total_draft, 1)
    return tokens[:n_tokens], acc_rate, total_accepted, total_draft, cycle_count


def main():
    print("Loading models...")
    target_params, target_config = load_model("weights.pkl")  # d=768
    draft_params, draft_config = load_model("weights_ctx2048.pkl")  # d=512 GQA
    vocab_size = target_config["vocab_size"]

    print(f"Draft:  d={draft_config['d_model']}, h={draft_config['n_heads']}, "
          f"kv_h={draft_config.get('n_kv_heads', draft_config['n_heads'])}, "
          f"l={draft_config['n_layers']}, ctx={draft_config['context_len']}")
    print(f"Target: d={target_config['d_model']}, h={target_config['n_heads']}, "
          f"kv_h={target_config.get('n_kv_heads', target_config['n_heads'])}, "
          f"l={target_config['n_layers']}, ctx={target_config['context_len']}")

    # Use a variety of prompts to get representative acceptance rates
    prompts = [
        [1, 42, 100, 7, 255],
        [50, 200, 3, 88, 1024, 512],
        [10, 20, 30, 40, 50, 60, 70],
        [999, 1, 500, 250, 125],
    ]

    N_TOKENS = 128

    # Warmup both models
    print("\nWarming up...", end=" ", flush=True)
    t_w, t_first, t_plen, t_kv = prefill(target_params, target_config, prompts[0], vocab_size)
    _ = int(multi_sm_decode_nlayer(t_w, target_config, t_first, t_plen, t_kv, vocab_size)[0])
    d_w, d_first, d_plen, d_kv = prefill(draft_params, draft_config, prompts[0], vocab_size)
    _ = int(multi_sm_decode_nlayer(d_w, draft_config, d_first, d_plen, d_kv, vocab_size)[0])
    print("done")

    # First: measure acceptance rates across prompts and K values
    print(f"\n{'='*60}")
    print(f"ACCEPTANCE RATE BENCHMARK ({N_TOKENS} tokens per prompt)")
    print(f"{'='*60}")

    for K in [2, 4, 6, 8]:
        total_acc = 0
        total_draft = 0
        total_cycles = 0

        for prompt_ids in prompts:
            t_w, t_first, t_plen, t_kv = prefill(target_params, target_config, prompt_ids, vocab_size)
            d_w, d_first, d_plen, d_kv = prefill(draft_params, draft_config, prompt_ids, vocab_size)

            _, acc_rate, n_acc, n_draft, n_cycles = generate_speculative(
                d_w, draft_config, t_w, target_config,
                d_first, t_first, t_plen,
                d_kv, t_kv, vocab_size, N_TOKENS, K=K)

            total_acc += n_acc
            total_draft += n_draft
            total_cycles += n_cycles

        overall_acc = total_acc / max(total_draft, 1)
        avg_accepted = total_acc / max(total_cycles, 1)
        print(f"  K={K}: acceptance={overall_acc:.1%}, "
              f"avg accepted/cycle={avg_accepted:.1f}/{K}, "
              f"effective tokens/cycle={avg_accepted + 1:.1f}")

    # Throughput benchmark
    print(f"\n{'='*60}")
    print(f"THROUGHPUT BENCHMARK (K=4, {N_TOKENS} tokens)")
    print(f"{'='*60}")

    prompt_ids = prompts[0]

    # Standard target decode
    t_w, t_first, t_plen, t_kv = prefill(target_params, target_config, prompt_ids, vocab_size)
    t0 = time.perf_counter()
    std_tokens, _ = generate_standard(t_w, target_config, t_first, t_plen, t_kv, vocab_size, N_TOKENS)
    t_std = time.perf_counter() - t0
    std_tps = N_TOKENS / t_std
    print(f"  Standard (d=768):    {std_tps:.0f} tok/s ({t_std*1000:.0f}ms)")

    # Speculative K=4
    t_w, t_first, t_plen, t_kv = prefill(target_params, target_config, prompt_ids, vocab_size)
    d_w, d_first, d_plen, d_kv = prefill(draft_params, draft_config, prompt_ids, vocab_size)
    t0 = time.perf_counter()
    spec_tokens, acc_rate, _, _, _ = generate_speculative(
        d_w, draft_config, t_w, target_config,
        d_first, t_first, t_plen,
        d_kv, t_kv, vocab_size, N_TOKENS, K=4)
    t_spec = time.perf_counter() - t0
    spec_tps = N_TOKENS / t_spec
    print(f"  Speculative K=4:     {spec_tps:.0f} tok/s ({t_spec*1000:.0f}ms), "
          f"acceptance={acc_rate:.1%}")
    print(f"  Speedup:             {spec_tps/std_tps:.2f}x")

    # Verify text matches
    text_match = std_tokens == spec_tokens
    if not text_match:
        # Find first difference
        for i, (a, b) in enumerate(zip(std_tokens, spec_tokens)):
            if a != b:
                print(f"  Text diverges at token {i}: std={a}, spec={b}")
                break
    print(f"  Text match:          {'YES' if text_match else 'NO (expected — different decode order)'}")

    # Analysis
    print(f"\n{'='*60}")
    print(f"ANALYSIS")
    print(f"{'='*60}")
    print(f"  Sequential verify is {spec_tps/std_tps:.2f}x vs standard.")
    print(f"  For parallel verify to break even at K=4, {acc_rate:.0%} acceptance:")
    t_draft_step = 0.55  # ms, d=512 sync'd
    t_target_step = 1.0  # ms, d=768 sync'd
    avg_tokens = 4 * acc_rate + 1
    needed_verify = avg_tokens * t_target_step - 4 * t_draft_step
    print(f"    Draft cost:  {4 * t_draft_step:.1f}ms")
    print(f"    Verify budget: {needed_verify:.1f}ms (to match {std_tps:.0f} tok/s)")
    print(f"    i.e., parallel verify must take < {needed_verify:.1f}ms for K=4 tokens")

    # Save results
    results = {
        "draft": f"d={draft_config['d_model']}, l={draft_config['n_layers']}",
        "target": f"d={target_config['d_model']}, l={target_config['n_layers']}",
        "std_tps": std_tps,
        "spec_tps": spec_tps,
        "acceptance": acc_rate,
    }
    with open("speculative_results.txt", "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print("\nResults saved to speculative_results.txt")


if __name__ == "__main__":
    main()
