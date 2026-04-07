# Data Research: Continued Training Plan for 306M Transformer

*April 2026 — Research findings for next training phase*

---

## Current State

```
Model: 306M params, d=1024, h=16, kv=4, l=24, ctx=512, vocab=32K
Training: 3 epochs × 7.85B unique tokens = 23.5B tokens seen
Tokens/param ratio: 25.6x (unique), 76.8x (total)
val_loss=2.86, ppl=17.42

Current data mix (7.64B train tokens):
  35.5%  FineWeb-Edu (score >= 3)       2.72B
  28.6%  StarCoder code (13 languages)  2.18B
  19.2%  OpenWebMath                    1.47B
   9.0%  Wikipedia                      0.69B
   8.1%  Cosmopedia (synthetic)         0.62B
```

---

## Key Research Findings

### 1. Scaling Laws: We Are Massively Undertrained

**Chinchilla (Hoffmann et al., 2022):** compute-optimal is ~20 tokens/param = 6.1B for 306M.
We've passed this at 23.5B total, but Chinchilla optimizes *training compute*, not
*inference quality*. Modern practice intentionally overtrains small models.

**Modern tokens/param ratios:**

| Model | Params | Tokens | Tokens/Param |
|-------|--------|--------|-------------|
| Chinchilla-optimal | 306M | 6.1B | 20x |
| **Ours (current)** | **306M** | **7.85B unique** | **25.6x** |
| LLaMA-1 7B | 7B | 1T | 143x |
| MiniCPM recommended | 306M | 59B | 192x |
| LLaMA-2 7B | 7B | 2T | 286x |
| TinyLlama 1.1B | 1.1B | 3T | 2,727x |
| LLaMA-3.1 8B | 8B | 15T | 1,875x |
| SmolLM2 360M | 360M | 4T | 11,111x |

Our 25.6x unique tokens/param is **7.5x below** even MiniCPM's conservative recommendation.
SmolLM2-360M trained on 430x more tokens per parameter than us.

**Conclusion:** More unique data is by far the biggest lever. Gains from better mix or
training tricks are secondary to simply seeing more diverse text.

### 2. Data Repetition: 3 Epochs Is Fine, But Not the Answer

**Muennighoff et al. (2023) "Scaling Data-Constrained Language Models":**
- Up to 4 epochs: negligible loss degradation
- Beyond 4: diminishing returns, eventually harmful
- Smaller models (175M-class) tolerate up to ~16 epochs
- Larger models (8B+) show harm after 4 epochs

Our 306M at 3 epochs is safely in the green zone. A 4th epoch is the last "free" one.
But more repetition is not the path — more unique data is.

### 3. Data Mix: Too Much Code and Math, Not Enough Web

**Comparison with peer models:**

| Model | Web | Code | Math | Books/Wiki | Synthetic |
|-------|-----|------|------|------------|-----------|
| **Ours** | **44%** | **29%** | **19%** | **9%** | **8%** |
| SmolLM2-360M | 70-75% | 15-20% | 5-10% | — | 5% |
| OLMo 1B | 78% | 8% | — | 6% | — |
| TinyLlama 1.1B | 87% | 13% | — | — | — |
| Llama 3 | 50% | 25% | 8% | 17% | — |

**Our web text is 44% vs 70-80% in peer models.** This hurts general language understanding.
Code (29%) and math (19%) are 2-3x higher than typical general-purpose mixes.

**However:** Phi models show that overweighting structured data (code + math + synthetic)
can boost reasoning benchmarks. If the goal is general-purpose, rebalance toward web.
If reasoning-focused, the current mix is defensible.

**Recommendation:** shift to ~55% web, 20% code, 12% math, 8% wiki, 5% synthetic.
This is a compromise between general-purpose and reasoning quality.

### 4. Multi-Phase Training: WSD + Annealing

**WSD Schedule (MiniCPM, 2024):** Warmup-Stable-Decay.
- Warmup: 1-2% of steps, LR ramps to peak
- Stable: 60-80% of steps, LR stays at peak
- Decay: 10-20% of steps, LR decays exponentially to 0

Key advantage: you can "branch off" from the stable phase at any point by starting
decay. Don't need to commit total compute budget upfront.

**Annealing data (Llama 3, SmolLM2):**
- Llama 3: anneals LR to 0 over final 40M tokens, upsamples highest-quality data
- SmolLM2: introduces highest-quality datasets during annealing
  (FineMath 4+, InfiWebMath 3+, Stack-Edu)

**Recommendation:** After main pretraining, run annealing phase of ~2-3B tokens
on highest-quality subsets: FineWeb-Edu score >= 4, FineMath 4+, Stack-Edu.
Linear LR decay from last stable LR to 0.

### 5. Context Length: Extend to 1024 After Pretraining

- RoPE with base=10000 generalizes fine to 2x training length (no adjustment needed)
- VRAM at ctx=1024, bs=8: estimated ~13-14GB (feasible on 16GB)
- Alternatively: ctx=1024, bs=16 with gradient checkpointing (--checkpoint)
- Should be a short final phase (~1-2B tokens of long documents)

### 6. Instruction Tuning: Skip at This Scale

FLAN paper (Chung et al.) found instruction tuning *degrades* performance at small
scales (<1B params). The model memorizes instruction patterns rather than generalizing.
SmolLM2-360M-Instruct only works because it was pretrained on 4T tokens first.
Skip until we've done much more pretraining.

---

## Recommended Training Plan

### Phase 1: Extended Pretraining (v3 dataset)

**New data mix (~28B unique tokens):**

```
Source                  Tokens    Pct    Notes
FineWeb-Edu (>= 3)     10.0B    36%    Expand from 2.72B, use more of sample-10BT
DCLM-Edu                5.0B    18%    NEW: quality-filtered web from DataComp
StarCoder               5.5B    20%    Expand from 2.18B, more languages
OpenWebMath             3.0B    11%    Expand from 1.47B, use more of full dataset
Wikipedia               2.0B     7%    Expand from 0.69B, full English wiki
Cosmopedia v2           2.5B     9%    Expand from 0.62B, synthetic textbooks
─────────────────────────────────────
Total                  28.0B   100%
```

**Why these numbers:**
- Web (FineWeb-Edu + DCLM-Edu + Wikipedia + Cosmopedia): 70% — matches SmolLM2
- Code (StarCoder): 20% — between Llama3 (25%) and SmolLM2 (15-20%)
- Math (OpenWebMath): 11% — reduced from 19% but still above most peers (reasoning focus)
- 28B tokens = ~91 tokens/param. With 2 epochs = ~183 tokens/param (near MiniCPM's 192x)

**Training schedule:**
- Use WSD: warmup 2K steps, stable at LR 3e-4, decay last 15%
- Curriculum: ctx=128 (10%), ctx=256 (20%), ctx=512 (70%)
- 2 epochs on v3 dataset = ~56B tokens total
- Estimated time: 2 epochs × ~28B tokens / 28.4K tok/s ≈ 548h on RTX 4080 Super
  (or ~40h on B200)

### Phase 2: Annealing (high-quality cooldown)

**~3B tokens of highest-quality data:**
```
FineWeb-Edu score >= 4    1.5B    50%
FineMath 4+               0.8B    27%
Stack-Edu                 0.7B    23%
```

- Linear LR decay from last stable LR to 0
- ctx=512, same batch size
- Estimated time: ~29h on 4080 Super

### Phase 3: Context Extension (optional)

- Short training at ctx=1024, bs=8 (or bs=16 with gradient checkpointing)
- ~1-2B tokens of long documents (filtered for length >= 1024 tokens)
- LR: 1e-4 (low), cosine decay
- Estimated time: ~20h on 4080 Super

---

## Data Sources on HuggingFace

| Source | HF Path | Size | Notes |
|--------|---------|------|-------|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` sample-10BT | ~10B tok | score >= 3, quality web |
| DCLM-Edu | `HuggingFaceTB/dclm-edu` | 100B+ tok | NEW: edu-filtered DCLM |
| StarCoder | `bigcode/starcoderdata` | 250B+ tok | 13 languages, deduplicated |
| OpenWebMath | `open-web-math/open-web-math` | ~14.7B tok | math with LaTeX |
| Wikipedia | `wikimedia/wikipedia` 20231101.en | ~3-4B tok | full English |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus` cosmopedia-v2 | ~30B tok | synthetic textbooks |
| FineMath | `HuggingFaceTB/finemath` | ~54B tok | step-by-step math (annealing) |
| Stack-Edu | `HuggingFaceTB/stack-edu` | ~125B tok | quality code (annealing) |

---

## What Changes in the Pipeline

1. **New script:** `prepare_data_v3.py` — downloads expanded sources + DCLM-Edu
2. **Output:** `data/tokens_v3/train.bin` + `val.npy` (~28B tokens, ~112GB)
3. **Existing data preserved:** `data/tokens_v2/` untouched
4. **Tokenizer:** same 32K BPE (no change)
5. **Annealing data:** separate `data/tokens_v3_anneal/` for phase 2

---

## References

- Hoffmann et al. (2022). "Training Compute-Optimal Large Language Models" (Chinchilla)
- Muennighoff et al. (2023). "Scaling Data-Constrained Language Models"
- Touvron et al. (2023). "LLaMA: Open and Efficient Foundation Language Models"
- Dubey et al. (2024). "The Llama 3 Herd of Models"
- Allal et al. (2025). "SmolLM2: When Smol Goes Big"
- Gu et al. (2024). "MiniCPM: Unveiling the Potential of Small Language Models"
- Li et al. (2023). "Textbooks Are All You Need" (Phi-1)
- Chung et al. (2022). "Scaling Instruction-Finetuned Language Models" (FLAN)
