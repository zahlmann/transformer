"""
validate.py — LOCKED. Do not modify.

Runs train_eggroll_triton.py with 3 seeds, checks locked constants,
reports pass/fail. Also runs backprop baseline for fair comparison.

Usage: uv run validate.py
"""
import subprocess
import sys
import re
import csv
import os
import time
from datetime import datetime

# These values are locked. If the training script prints different values the
# run counts as a CHEAT and is rejected regardless of val_loss.
REQUIRED = {
    "D_MODEL": 64,
    "N_HEADS": 2,
    "N_LAYERS": 1,
    "CONTEXT_LEN": 128,
    "BATCH_SIZE": 128,
    "EPOCHS": 10,
    "TEMPERATURE": 2.0,
}

SEEDS = [42, 11, 7]
BACKPROP_TARGET = 1.84   # backprop+Adam val_loss at 10 epochs (LR=3e-3, seed=42)
MAX_MEMORY_MB = 2000.0   # hard ceiling
RESULTS_TSV = "results.tsv"


def parse_block(output: str, block_name: str) -> dict:
    pattern = rf"=== {re.escape(block_name)} ===\n(.*?)==+\n"
    m = re.search(pattern, output, re.DOTALL)
    if not m:
        return {}
    result = {}
    for line in m.group(1).strip().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            result[k.strip()] = v.strip()
    return result


def run_seed(seed: int) -> dict | None:
    print(f"\n--- seed {seed} ---")
    result = subprocess.run(
        ["uv", "run", "train_eggroll_triton.py", "--seed", str(seed)],
        capture_output=True, text=True, timeout=1200,
    )

    combined = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"  CRASHED (exit {result.returncode})")
        print(result.stderr[-800:] if result.stderr else "")
        return None

    constants = parse_block(result.stdout, "CONSTANTS")
    results = parse_block(result.stdout, "RESULTS")

    if not constants or not results:
        print("  ERROR: could not parse CONSTANTS or RESULTS block")
        print(result.stdout[-500:])
        return None

    violations = []
    for key, expected in REQUIRED.items():
        actual_raw = constants.get(key)
        if actual_raw is None:
            violations.append(f"  {key} missing from CONSTANTS block")
            continue
        actual = float(actual_raw)
        if abs(actual - expected) > 1e-6:
            violations.append(f"  {key}: expected {expected}, got {actual}")
    if violations:
        print("  CHEAT DETECTED — locked constants were changed:")
        for v in violations:
            print(v)
        return None

    parsed = {
        "seed": seed,
        "val_loss": float(results["val_loss"]),
        "perplexity": float(results["perplexity"]),
        "training_time_s": float(results["training_time_s"]),
        "peak_memory_mb": float(results["peak_memory_mb"]),
    }
    print(f"  val_loss={parsed['val_loss']:.4f}  ppl={parsed['perplexity']:.2f}  "
          f"time={parsed['training_time_s']:.1f}s  mem={parsed['peak_memory_mb']:.0f}MB")
    return parsed


def write_tsv(row: dict):
    fieldnames = ["timestamp", "commit", "seed", "val_loss", "perplexity",
                  "training_time_s", "peak_memory_mb", "status", "description"]
    exists = os.path.exists(RESULTS_TSV)
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t",
                                extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip()
    except Exception:
        return "unknown"


def main():
    print("=" * 60)
    print("validate.py — 3-seed evaluation")
    print(f"Backprop target: val_loss <= {BACKPROP_TARGET}")
    print("=" * 60)

    commit = get_commit()
    runs = []
    for seed in SEEDS:
        r = run_seed(seed)
        if r is None:
            write_tsv({
                "timestamp": datetime.now().isoformat(),
                "commit": commit, "seed": seed,
                "val_loss": 0.0, "perplexity": 0.0,
                "training_time_s": 0.0, "peak_memory_mb": 0.0,
                "status": "crash", "description": "crashed or cheat detected",
            })
            print("\nFAIL — aborting validation")
            sys.exit(1)
        runs.append(r)

    avg_loss = sum(r["val_loss"] for r in runs) / len(runs)
    avg_ppl = sum(r["perplexity"] for r in runs) / len(runs)
    avg_time = sum(r["training_time_s"] for r in runs) / len(runs)
    max_mem = max(r["peak_memory_mb"] for r in runs)

    print("\n" + "=" * 60)
    print(f"  avg val_loss : {avg_loss:.4f}  (backprop target: {BACKPROP_TARGET})")
    print(f"  avg ppl      : {avg_ppl:.2f}")
    print(f"  avg time     : {avg_time:.1f}s")
    print(f"  max memory   : {max_mem:.0f}MB  (limit {MAX_MEMORY_MB:.0f}MB)")
    print(f"  gap to bp    : {avg_loss - BACKPROP_TARGET:+.4f}")
    print("=" * 60)

    mem_pass = max_mem <= MAX_MEMORY_MB
    loss_pass = avg_loss <= BACKPROP_TARGET

    if not mem_pass:
        print(f"FAIL — peak memory {max_mem:.0f}MB > {MAX_MEMORY_MB:.0f}MB")
    if loss_pass:
        print("PASS — EGGROLL matches backprop quality!")
    else:
        print(f"NOT YET — val_loss {avg_loss:.4f} > {BACKPROP_TARGET} (gap: {avg_loss - BACKPROP_TARGET:.4f})")

    status = "keep" if mem_pass else "discard"
    description = f"loss={avg_loss:.4f} ppl={avg_ppl:.2f} time={avg_time:.1f}s mem={max_mem:.0f}MB gap={avg_loss-BACKPROP_TARGET:+.4f}"

    for r in runs:
        write_tsv({
            "timestamp": datetime.now().isoformat(),
            "commit": commit, **r,
            "status": status, "description": description,
        })

    sys.exit(0 if mem_pass else 1)


if __name__ == "__main__":
    main()
