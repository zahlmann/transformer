# 306M Transformer

Training and inference for a 306M parameter transformer on a single GPU. Custom Triton kernels for inference, JAX + cuDNN for training.

Built using [karpathy/autoresearch](https://github.com/karpathy/autoresearch)-style autonomous development — a coding agent is pointed at `knowledge/program.md` repeatedly. The human steers direction; the agent handles implementation, debugging, benchmarking, and documentation. The agent provides insights about findings that the human then studies to build deep understanding.

## Documentation

- [`program.md`](knowledge/program.md) — the agent program that drove development, with full architecture decisions and optimization history
- [`inference_explained.md`](knowledge/inference_explained.md) — ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all inference techniques used
- [`training_explained.md`](knowledge/training_explained.md) — first-principles explanation of the training pipeline (data, model, training loop)
- [`data_research.md`](knowledge/data_research.md) — training data research and rationale

## Current Model

```
306M params (d=1024, h=16, kv=4, l=24, ctx=512)
  RMSNorm, RoPE, SwiGLU (d_ff=2816), GQA, no biases, tied embeddings
  Vocab: 32K BPE trained on corpus
  Training data: 7.85B tokens (5 sources)
    34% FineWeb-Edu    — quality-filtered web (score >= 3)
    30% StarCoder      — code (13 languages)
    19% OpenWebMath    — math with LaTeX
     9% Wikipedia
     8% Cosmopedia     — synthetic textbooks
```

## Performance (RTX 4080 Super)

```
Inference:
  Decode:    231 tok/s  (4.3 ms/tok, Triton multi-SM kernel)
  Prefill:   157 ms for 128 tokens (JAX)
  Weights:   607 MB bf16 (9.5x L2 — HBM-bound)

Training:
  RTX 4080 Super:  28.4K tok/s (bs=16), ~83h per epoch
  NVIDIA B200:     ~341K tok/s (bs=256), ~6h per epoch
  3 epochs x 7.85B tokens = 23.5B total
```

The entire decode step — embedding, attention, FFN, output projection — runs in a single GPU kernel call across all 24 layers.

## Quick Start

Requires `weights.pkl` in the repo root (not included — too large for git).

```bash
# generate text (streaming, with sampling)
uv run generate.py --prompt "Once upon a time" --temp 0.7 --top-p 0.95 --rep-penalty 1.2

# greedy decoding
uv run generate.py --prompt "The capital of France is"

# profile decode kernel
uv run profile_kernels.py

# prepare training data (download + tokenize 7.85B tokens from 5 sources)
uv run prepare_data_v2.py

# train (see knowledge/gpu_server_training_instructions.md for cloud GPU setup)
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 16 --epochs 3 \
  --curriculum --lr 3e-4 --no-checkpoint
```

## What's In Here

**Training**: JAX model with cuDNN FlashAttention, AdamW with cosine LR schedule, curriculum training (ctx warmup 128->256->512). Fused cross-entropy for 32K vocab without OOM. Multi-source data pipeline with 32K BPE tokenizer.

**Inference kernel**: Custom Triton kernel fuses the entire 24-layer forward pass (embedding, RMSNorm, RoPE, GQA attention, SwiGLU FFN, output projection, argmax) into a single GPU kernel launch. Multi-SM parallelism with atomic barriers across 16 SMs.

## Files

```
Training:
  model.py                             JAX transformer (RMSNorm, RoPE, SwiGLU, GQA, fused CE)
  train.py                             AdamW training (bf16 fwd, cuDNN FlashAttn, curriculum)
  data.py                              streaming data loading (v2/v3 memmap, 8B+ tokens)
  prepare_data_v2.py                   v2 data pipeline: 5-source download, tokenize, shuffle
  prepare_data_v3.py                   v3 data pipeline: 5 sources (~50B) + annealing (3B)

Inference:
  kernels/multi_sm_decode.py           fused multi-SM decode (all 24 layers in one kernel)
  kernels/fused_decode_nlayer.py       weight/KV packing + single-SM decode kernel
  generate.py                          streaming text generation CLI
  profile_kernels.py                   decode kernel profiling

Documentation (knowledge/):
  program.md                           agent program / development log
  inference_explained.md               ground-up GPU kernel explanation
  training_explained.md                first-principles training pipeline explanation
  data_research.md                     training data research findings
  gpu_server_training_instructions.md  cloud GPU training setup guide
```
