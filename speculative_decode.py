"""Speculative decoding benchmark: d=512 draft + d=768 target.

Measures acceptance rates and throughput. Sequential verification can't be
profitable (t_draft + t_verify > t_target), but acceptance rates show whether
a parallel verify kernel would be worthwhile.
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle, sys, time
import jax.numpy as jnp
import numpy as np
from model import prefill_with_kv
from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.multi_sm_decode import multi_sm_decode_nlayer


def load_model(path):
    with open(os.path.join(os.path.dirname(__file__), path), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    return params, saved["config"]


def prefill(params, config, prompt_ids):
    ctx_len = config["context_len"]
    prompt_len = len(prompt_ids)
    assert prompt_len < ctx_len
    vocab_size = config["vocab_size"]
    x = jnp.pad(jnp.array(prompt_ids, dtype=jnp.int32),
                 (0, ctx_len - prompt_len)).astype(jnp.int32)
    logits, k_caches, v_caches = prefill_with_kv(params, config, x)
    _ = logits.block_until_ready()
    w = prepare_decode_weights_nlayer(params, config, vocab_size)
    kv = pack_kv_caches(k_caches, v_caches)
    first_token = int(jnp.argmax(logits[prompt_len - 1]))
    return w, first_token, prompt_len, kv


def decode_step(w, config, tok, pos, kv):
    vocab_size = config["vocab_size"]
    tok_out, _, kv_out = multi_sm_decode_nlayer(w, config, tok, pos, kv, vocab_size)
    return int(tok_out), kv_out


def generate_standard(w, config, first_token, prompt_len, kv, n_tokens):
    tokens = [first_token]
    tok, kv_cur = first_token, kv
    for i in range(n_tokens - 1):
        tok, kv_cur = decode_step(w, config, tok, prompt_len + i, kv_cur)
        tokens.append(tok)
    return tokens


def generate_speculative(d_w, d_cfg, t_w, t_cfg, d_first, t_first,
                         prompt_len, d_kv, t_kv, n_tokens, K):
    tok = t_first
    tokens = [tok]
    pos = prompt_len
    total_draft, total_accepted = 0, 0

    while len(tokens) < n_tokens:
        k = min(K, n_tokens - len(tokens))

        # draft K tokens
        draft_tokens = []
        draft_tok, draft_kv_tmp = tok, d_kv
        for j in range(k):
            draft_tok, draft_kv_tmp = decode_step(d_w, d_cfg, draft_tok, pos + j, draft_kv_tmp)
            draft_tokens.append(draft_tok)
        total_draft += k

        # verify sequentially through target
        target_toks = []
        verify_tok, verify_kv = tok, t_kv
        for j in range(k):
            t_tok, verify_kv = decode_step(t_w, t_cfg, verify_tok, pos + j, verify_kv)
            target_toks.append(t_tok)
            verify_tok = draft_tokens[j]

        # find first disagreement
        n_acc = 0
        for j in range(k):
            if target_toks[j] != draft_tokens[j]:
                break
            n_acc += 1
        total_accepted += n_acc

        if n_acc == k:
            tokens.extend(draft_tokens)
            pos += k
            bonus, verify_kv = decode_step(t_w, t_cfg, draft_tokens[-1], pos, verify_kv)
            tokens.append(bonus)
            pos += 1
            _, draft_kv_tmp = decode_step(d_w, d_cfg, draft_tokens[-1], pos - 1, draft_kv_tmp)
            t_kv, d_kv, tok = verify_kv, draft_kv_tmp, bonus
        else:
            correction = target_toks[n_acc]
            tokens.extend(draft_tokens[:n_acc])
            tokens.append(correction)
            pos += n_acc + 1
            # re-advance draft through accepted + correction
            rb_kv, rb_tok = d_kv, tok
            for jj in range(n_acc + 1):
                actual = draft_tokens[jj] if jj < n_acc else correction
                _, rb_kv = decode_step(d_w, d_cfg, rb_tok, pos - n_acc - 1 + jj, rb_kv)
                rb_tok = actual
            t_kv, d_kv, tok = verify_kv, rb_kv, correction

    acc_rate = total_accepted / max(total_draft, 1)
    return tokens[:n_tokens], acc_rate


def main():
    print("Loading models...")
    t_params, t_cfg = load_model("weights.pkl")       # d=768 target
    d_params, d_cfg = load_model("weights_ctx2048.pkl")  # d=512 draft
    assert t_cfg["vocab_size"] == d_cfg["vocab_size"]

    print(f"Draft:  d={d_cfg['d_model']}, l={d_cfg['n_layers']}")
    print(f"Target: d={t_cfg['d_model']}, l={t_cfg['n_layers']}")

    prompts = [[1, 42, 100, 7, 255], [50, 200, 3, 88, 1024, 512],
               [10, 20, 30, 40, 50, 60, 70], [999, 1, 500, 250, 125]]
    N = 128

    # warmup
    t_w, t_first, t_plen, t_kv = prefill(t_params, t_cfg, prompts[0])
    _ = decode_step(t_w, t_cfg, t_first, t_plen, t_kv)
    d_w, d_first, d_plen, d_kv = prefill(d_params, d_cfg, prompts[0])
    _ = decode_step(d_w, d_cfg, d_first, d_plen, d_kv)

    # acceptance rates
    print(f"\n=== Acceptance Rates ({N} tokens, {len(prompts)} prompts) ===")
    for K in [2, 4, 6, 8]:
        total_acc, total_draft = 0, 0
        for p in prompts:
            t_w, t_first, t_plen, t_kv = prefill(t_params, t_cfg, p)
            d_w, d_first, d_plen, d_kv = prefill(d_params, d_cfg, p)
            _, acc = generate_speculative(d_w, d_cfg, t_w, t_cfg,
                                          d_first, t_first, t_plen, d_kv, t_kv, N, K)
            total_acc += int(acc * N)  # approximate
        avg_acc = total_acc / (len(prompts) * N)
        print(f"  K={K}: ~{avg_acc:.0%} acceptance")

    # throughput
    print(f"\n=== Throughput (K=4) ===")
    p = prompts[0]
    t_w, t_first, t_plen, t_kv = prefill(t_params, t_cfg, p)
    t0 = time.perf_counter()
    std_tokens = generate_standard(t_w, t_cfg, t_first, t_plen, t_kv, N)
    t_std = time.perf_counter() - t0

    t_w, t_first, t_plen, t_kv = prefill(t_params, t_cfg, p)
    d_w, d_first, d_plen, d_kv = prefill(d_params, d_cfg, p)
    t0 = time.perf_counter()
    spec_tokens, acc = generate_speculative(d_w, d_cfg, t_w, t_cfg,
                                            d_first, t_first, t_plen, d_kv, t_kv, N, K=4)
    t_spec = time.perf_counter() - t0

    std_tps = N / t_std
    spec_tps = N / t_spec
    print(f"  Standard:   {std_tps:.0f} tok/s")
    print(f"  Spec K=4:   {spec_tps:.0f} tok/s ({spec_tps/std_tps:.2f}x), acceptance={acc:.0%}")
    print(f"  Text match: {std_tokens == spec_tokens}")


if __name__ == "__main__":
    main()
