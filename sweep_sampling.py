"""Sweep sampling parameters across domains to find best settings.

Tests temperature, top-p, and repetition penalty combinations on
story, knowledge, code, and creative prompts. Saves results to
sweep_results.txt for comparison.

Usage: uv run sweep_sampling.py
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import pickle
import time
import numpy as np
import jax.numpy as jnp

from data import load_bpe_vocab
from generate import stream_tokens

PROMPTS = {
    "story": "Once upon a time, a little girl named Lily",
    "knowledge": "The solar system consists of",
    "code": "def fibonacci(n):",
    "creative": "In a world where machines could dream,",
    "explanation": "Machine learning is a field of",
    "dialogue": "\"Hello,\" said the old man. \"I've been waiting for you.\"",
}

SETTINGS = [
    # (temp, top_p, rep_penalty, label)
    (0.0,  1.0,  1.0,  "greedy"),
    (0.5,  1.0,  1.0,  "temp=0.5"),
    (0.7,  1.0,  1.0,  "temp=0.7"),
    (0.9,  1.0,  1.0,  "temp=0.9"),
    (1.0,  1.0,  1.0,  "temp=1.0"),
    (0.7,  0.9,  1.0,  "temp=0.7 top_p=0.9"),
    (0.7,  0.95, 1.0,  "temp=0.7 top_p=0.95"),
    (0.8,  0.9,  1.0,  "temp=0.8 top_p=0.9"),
    (0.8,  0.95, 1.0,  "temp=0.8 top_p=0.95"),
    (0.9,  0.95, 1.0,  "temp=0.9 top_p=0.95"),
    (0.7,  0.95, 1.1,  "temp=0.7 top_p=0.95 rep=1.1"),
    (0.7,  0.95, 1.2,  "temp=0.7 top_p=0.95 rep=1.2"),
    (0.7,  0.95, 1.3,  "temp=0.7 top_p=0.95 rep=1.3"),
    (0.8,  0.95, 1.1,  "temp=0.8 top_p=0.95 rep=1.1"),
    (0.8,  0.95, 1.2,  "temp=0.8 top_p=0.95 rep=1.2"),
    (0.8,  0.95, 1.3,  "temp=0.8 top_p=0.95 rep=1.3"),
    (0.9,  0.95, 1.2,  "temp=0.9 top_p=0.95 rep=1.2"),
    (1.0,  0.95, 1.2,  "temp=1.0 top_p=0.95 rep=1.2"),
]

MAX_TOKENS = 200
SEED = 42


def repetition_score(token_ids):
    """Measure repetition: fraction of unique 4-grams out of total 4-grams."""
    if len(token_ids) < 4:
        return 1.0
    ngrams = [tuple(token_ids[i:i+4]) for i in range(len(token_ids) - 3)]
    return len(set(ngrams)) / len(ngrams)


def main():
    # Load model once
    print("Loading model...")
    with open(os.path.join(os.path.dirname(__file__), "weights.pkl"), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]

    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(bpe_vocab["tokenizer_path"])
    encode_fn = lambda text: tok.encode(text).ids

    d = config["d_model"]
    n_layers = config["n_layers"]
    print(f"Model: d={d}, l={n_layers}, {sum(p.size for p in params.values()):,} params")

    out_path = os.path.join(os.path.dirname(__file__), "sweep_results.txt")
    results = []

    total_runs = len(SETTINGS) * len(PROMPTS)
    run_i = 0

    for temp, top_p, rep_penalty, label in SETTINGS:
        print(f"\n{'='*80}")
        print(f"Settings: {label}")
        print(f"{'='*80}")

        setting_scores = []

        for domain, prompt_text in PROMPTS.items():
            run_i += 1
            prompt_ids = jnp.array(encode_fn(prompt_text), dtype=jnp.int32)
            max_gen = min(MAX_TOKENS, config["context_len"] - len(prompt_ids))

            np.random.seed(SEED)
            t0 = time.perf_counter()
            tokens = list(stream_tokens(
                params, config, prompt_ids, max_gen,
                temperature=temp, top_p=top_p, rep_penalty=rep_penalty, seed=SEED))
            elapsed = time.perf_counter() - t0

            text = decode_fn(tokens)
            rep_score = repetition_score(tokens)
            tok_per_s = len(tokens) / elapsed

            print(f"\n  [{domain}] rep_diversity={rep_score:.2f} ({tok_per_s:.0f} tok/s)")
            print(f"  {prompt_text}{text[:300]}")

            setting_scores.append(rep_score)
            results.append({
                "label": label,
                "domain": domain,
                "prompt": prompt_text,
                "output": text[:500],
                "rep_diversity": rep_score,
                "tok_per_s": tok_per_s,
                "n_tokens": len(tokens),
            })

            print(f"  [{run_i}/{total_runs}]", end="", flush=True)

        avg_div = np.mean(setting_scores)
        print(f"\n  Average 4-gram diversity: {avg_div:.3f}")

    # Write results
    print(f"\n\nWriting results to {out_path}...")
    with open(out_path, "w") as f:
        f.write("Sampling Parameter Sweep Results\n")
        f.write(f"Model: d={config['d_model']}, l={config['n_layers']}, "
                f"{sum(p.size for p in params.values()):,} params\n")
        f.write(f"Max tokens: {MAX_TOKENS}, Seed: {SEED}\n")
        f.write(f"{'='*100}\n\n")

        # Summary table
        f.write("SUMMARY (average 4-gram diversity across all domains, 1.0 = no repetition)\n")
        f.write(f"{'Settings':<40} {'story':>8} {'knowl':>8} {'code':>8} {'creat':>8} "
                f"{'expl':>8} {'dial':>8} {'AVG':>8}\n")
        f.write("-" * 100 + "\n")

        for temp, top_p, rep_penalty, label in SETTINGS:
            row = [r for r in results if r["label"] == label]
            scores = {r["domain"]: r["rep_diversity"] for r in row}
            avg = np.mean(list(scores.values()))
            f.write(f"{label:<40} "
                    f"{scores.get('story', 0):>8.3f} "
                    f"{scores.get('knowledge', 0):>8.3f} "
                    f"{scores.get('code', 0):>8.3f} "
                    f"{scores.get('creative', 0):>8.3f} "
                    f"{scores.get('explanation', 0):>8.3f} "
                    f"{scores.get('dialogue', 0):>8.3f} "
                    f"{avg:>8.3f}\n")

        f.write(f"\n{'='*100}\n\n")
        f.write("FULL OUTPUTS\n\n")

        for r in results:
            f.write(f"--- {r['label']} | {r['domain']} | div={r['rep_diversity']:.3f} ---\n")
            f.write(f"Prompt: {r['prompt']}\n")
            f.write(f"Output: {r['output']}\n\n")

    print(f"Done! Results saved to {out_path}")

    # Print top 5 settings by average diversity
    print("\nTop 5 settings by average 4-gram diversity:")
    setting_avgs = []
    for temp, top_p, rep_penalty, label in SETTINGS:
        row = [r for r in results if r["label"] == label]
        avg = np.mean([r["rep_diversity"] for r in row])
        setting_avgs.append((label, avg))
    setting_avgs.sort(key=lambda x: -x[1])
    for label, avg in setting_avgs[:5]:
        print(f"  {label:<40} avg_diversity={avg:.3f}")


if __name__ == "__main__":
    main()
