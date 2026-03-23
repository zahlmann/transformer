"""Benchmark: run both backprop and EGGROLL training, compare results."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import subprocess


def run_script(name, script):
    print(f"\n{'='*60}")
    print(f"Running {name}...")
    print(f"{'='*60}\n")

    start = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", "python", script],
        capture_output=True, text=True, timeout=1200,
    )
    elapsed = time.perf_counter() - start

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])

    return result.returncode, elapsed


def main():
    print("EGGROLL Transformer Benchmark")
    print(f"{'='*60}")

    rc_bp, t_bp = run_script("Backprop Baseline", "train_backprop.py")
    rc_es, t_es = run_script("EGGROLL ES", "train_eggroll.py")

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Backprop: exit={rc_bp}, wall_time={t_bp:.1f}s")
    print(f"EGGROLL:  exit={rc_es}, wall_time={t_es:.1f}s")
    print(f"Speed ratio: {t_es/t_bp:.1f}x")

    # print results.tsv
    if os.path.exists("results.tsv"):
        print(f"\n{'='*60}")
        print("Results log (results.tsv)")
        print(f"{'='*60}")
        with open("results.tsv") as f:
            print(f.read())


if __name__ == "__main__":
    main()
