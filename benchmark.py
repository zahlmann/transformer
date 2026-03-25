"""
benchmark.py — fast single-seed run for development iteration.

Runs both backprop and EGGROLL at seed=42, compares results.
Use this during development for quick feedback before running validate.py.

Usage: uv run benchmark.py
"""
import subprocess
import sys
import re
import time

BACKPROP_BASELINE = {
    "val_loss": 1.84,    # backprop+Adam with LR=3e-3, seed=42, 10 epochs
    "training_time_s": 4.1,
}


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


def main():
    print("=" * 60)
    print("EGGROLL vs Backprop benchmark (seed=42, 10 epochs)")
    print("=" * 60)

    # Run EGGROLL
    print("\n--- Running EGGROLL ---")
    result = subprocess.run(
        ["uv", "run", "train_eggroll_triton.py", "--seed", "42"],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("EGGROLL CRASHED")
        print(result.stderr[-800:])
        sys.exit(1)

    # Print epoch-by-epoch progress
    for line in result.stdout.splitlines():
        if "Epoch" in line or "RESULTS" in line or "val_loss" in line:
            print(line)

    eggroll = parse_block(result.stdout, "RESULTS")
    if not eggroll:
        print("ERROR: could not parse RESULTS from EGGROLL output")
        sys.exit(1)

    eg_loss = float(eggroll["val_loss"])
    eg_ppl = float(eggroll["perplexity"])
    eg_time = float(eggroll["training_time_s"])
    eg_mem = float(eggroll.get("peak_memory_mb", 0))

    bp_loss = BACKPROP_BASELINE["val_loss"]
    bp_time = BACKPROP_BASELINE["training_time_s"]

    gap = eg_loss - bp_loss
    time_ratio = eg_time / bp_time

    print("\n" + "=" * 60)
    print("  COMPARISON (10 epochs, same architecture)")
    print(f"  {'':20s} {'EGGROLL':>10s} {'Backprop':>10s} {'Gap':>10s}")
    print(f"  {'val_loss':20s} {eg_loss:10.4f} {bp_loss:10.4f} {gap:+10.4f}")
    print(f"  {'perplexity':20s} {eg_ppl:10.2f} {float(2.718**bp_loss):10.2f}")
    print(f"  {'time (s)':20s} {eg_time:10.1f} {bp_time:10.1f} {time_ratio:10.0f}x")
    print(f"  {'memory (MB)':20s} {eg_mem:10.0f}")
    print("=" * 60)

    if gap <= 0:
        print("GOAL REACHED: EGGROLL matches or beats backprop!")
    else:
        print(f"Gap remaining: {gap:.4f} ({gap/bp_loss*100:.1f}% of backprop loss)")


if __name__ == "__main__":
    main()
