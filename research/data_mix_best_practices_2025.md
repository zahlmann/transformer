# Training Data Mix Best Practices (2025-2026)

## 1. Data Mix Ratios

**SmolLM2** (135M/360M/1.7B) final stage: **58% web, 24% code, 14% math, 4% synthetic textbooks** (Cosmopedia v2). Web split uses higher DCLM-to-FineWeb-Edu ratio.

**Llama 3**: **50% general web, 25% math/reasoning, 17% code, 8% multilingual**. Trained on 15T tokens total.

**Qwen 2.5-Coder** ablations found **70% code, 20% math, 10% general text** optimal for code-focused models. Qwen3 uses 36T tokens across 119 languages with heavy synthetic math/code augmentation.

**Phi-4**: 5T tokens, heavily synthetic. Exact ratios not published, but each domain (math, code) is optimized independently then merged additively.

**Consensus for a balanced small model**: ~55-60% web, 20-25% code, 12-15% math, 3-5% synthetic/instruction.

## 2. Data Quantity

Chinchilla (2022): 20 tokens per parameter. **Current practice (2025): ~300 tokens per parameter**, growing 3.1x/year.

Concrete numbers for small models:
- SmolLM2-135M: **2T tokens** (14,800 tok/param)
- SmolLM2-360M: **4T tokens** (11,100 tok/param)
- SmolLM2-1.7B: **11T tokens** (6,500 tok/param)

**For a 300M model: target 2-4T tokens.** This is 20x beyond Chinchilla-optimal (6B), but matches current SOTA practice. The SmolLM2-360M trained on 4T tokens is the closest reference point.

## 3. Data Repetition

SmolLM2 trained for ~2 epochs over collected datasets. StarCoderData was limited to 10% of mix to keep it at ~4 epochs over 11T tokens. Key findings:

- **Data quality > quantity**: optimally mixed 1B-token datasets outperform naive 10B-token datasets.
- **Moderate repetition is fine**: 2-4 epochs on high-quality data works well.
- **Quality-aware scaling laws** (2025-2026): filtering training data to remove noise effectively improves scaling behavior—a billion high-quality tokens >> a billion noisy tokens.

## 4. Best Open Math Pretraining Datasets

| Dataset | Tokens | Notes |
|---------|--------|-------|
| **Nemotron-CC-Math** | 133B (3+), 52B (4+) | NVIDIA, 2025. Best current option. Preserves equations/code, LaTeX standardized |
| **FineMath** | 34B (3+), 9.6B (4+) | HuggingFace, 2024. Strong GSM8K/MATH gains |
| **OpenWebMath** | 14.7B | 130K+ domains, faithful math notation |
| **ProofPile-2** | ~55B | Includes OpenWebMath + arXiv + algebraic code |

**Recommendation**: Nemotron-CC-Math-4+ (52B tokens) or FineMath-3+ (34B) as primary math source.

## 5. Instruction Data in Pretraining

Current best practice: **yes, mix instruction data during pretraining**, not just fine-tuning.

- SmolLM2 includes SmolTalk (instruction data) in later training stages during pretraining.
- Research shows a 500M model trained on instruction-augmented corpus (100B tokens) matches a 1B model trained on 3x more plain text.
- Phi-4 uses synthetic instruction-style data throughout pretraining.
- **Typical fraction**: 3-5% of total mix as instruction/conversation data, introduced in later training stages (not from the start).

## 6. EOS / Document Boundary Handling

Three requirements for packed sequences (ACL 2025, HuggingFace):

1. **Insert EOS token** between every document: `doc1 <eos> doc2 <eos> doc3`
2. **Mask cross-document attention** (critical): tokens must not attend across document boundaries. Use a block-diagonal causal mask.
3. **Reset position IDs** at each document boundary: each document starts at position 0.

All three are necessary. Without cross-document attention masking, the model attends to unrelated prior documents, hurting quality. Modern implementations (FlexAttention, sequence packing) handle this automatically.

---

Sources:
- [SmolLM2 paper](https://arxiv.org/abs/2502.02737)
- [Llama 3 breakdown](https://arize.com/blog/breaking-down-meta-llama-3/)
- [Qwen2.5-Coder report](https://arxiv.org/html/2409.12186v1)
- [Phi-4 report](https://arxiv.org/html/2412.08905v1)
- [FineMath dataset](https://huggingface.co/datasets/HuggingFaceTB/finemath)
- [Nemotron-CC-Math](https://huggingface.co/blog/nvidia/nemotron-cc-math)
- [Tokens per parameter trend](https://epoch.ai/data-insights/training-tokens-per-parameter)
- [Data Mixing Laws (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/cc84bfabe6389d8883fc2071c848f62a-Paper-Conference.pdf)
- [Sequence packing & masked attention](https://huggingface.co/blog/sirluk/llm-sequence-packing)
- [Segment-Based Attention Masking (ACL 2025)](https://aclanthology.org/2025.acl-long.947/)
