"""Fine-grained hyperparameter sweep around best config."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

from sweep_hp import run_config

configs = [
    # (sigma, lr, lr_decay, momentum, half_pop, label)
    (0.03, 0.020, 0.95, 0.5, 4096, "sigma=0.03 (best from sweep1)"),
    (0.03, 0.020, 0.95, 0.6, 4096, "sigma=0.03 mom=0.6"),
    (0.03, 0.025, 0.95, 0.5, 4096, "sigma=0.03 LR=0.025"),
    (0.025, 0.020, 0.95, 0.5, 4096, "sigma=0.025"),
    (0.02, 0.020, 0.95, 0.5, 4096, "sigma=0.02"),
    (0.03, 0.020, 0.93, 0.5, 4096, "sigma=0.03 decay=0.93"),
    (0.03, 0.020, 0.95, 0.5, 8192, "sigma=0.03 pop=16K"),
    (0.03, 0.015, 0.95, 0.6, 4096, "sigma=0.03 mom=0.6 LR=0.015"),
]

results = []
for sigma, lr, decay, mom, hp, label in configs:
    vl, t = run_config(sigma, lr, decay, mom, hp)
    print(f"  {label:40s}  val_loss={vl:.4f}  time={t:.0f}s")
    results.append((label, vl, t))

print("\n=== SWEEP 2 RESULTS ===")
results.sort(key=lambda x: x[1])
for label, vl, t in results:
    gap = vl - 2.45
    print(f"  {vl:.4f} ({gap:+.4f})  {t:6.0f}s  {label}")
