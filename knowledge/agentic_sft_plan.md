# Agentic SFT Data Generation Plan

Generate tool-use training traces from DeepSeek V3.2 via its API, then fine-tune
our 303M model to become an agentic coding/math assistant.

---

## Overview

```
Seed prompts (HuggingFace datasets)
    ↓
Evol-Instruct variations (3x expansion)
    ↓
DeepSeek V3.2 agent harness (function calling API)
    ↓  model generates tool calls, harness executes them, feeds results back
    ↓  full conversation trace recorded
Filter (keep only traces where final answer is correct)
    ↓
Format as SFT training data (mask everything except model outputs)
    ↓
Fine-tune 303M model
```

---

## 1. Seed Prompts

Download from HuggingFace. All open-licensed with ground truth for verification.

### Coding (~15K seeds)

```python
from datasets import load_dataset

# 10K coding competition problems with test cases
apps = load_dataset("codeparrot/apps", split="train")

# 1K basic Python tasks with assertions
mbpp = load_dataset("google-research-datasets/mbpp", split="train")

# 164 function-completion problems with unit tests
human_eval = load_dataset("openai/openai_humaneval", split="test")

# ~3K competitive programming problems
code_contests = load_dataset("deepmind/code_contests", split="train")
```

### Math (~21K seeds)

```python
# 8.5K grade school word problems with numeric answers
gsm8k = load_dataset("openai/gsm8k", split="train")

# 12.5K competition math with answers
math = load_dataset("hendrycks/competition_math", split="train")
```

### Total: ~36K seed prompts before augmentation

---

## 2. Prompt Augmentation (Evol-Instruct)

Expand 36K seeds to ~100K diverse prompts. Use DeepSeek to mutate each seed:

```python
MUTATIONS = [
    "Make this problem harder by adding an edge case constraint.",
    "Rewrite this as a multi-step task requiring intermediate computation.",
    "Add a requirement to handle errors gracefully.",
    "Change the domain but keep the same algorithmic structure.",
    "Turn this into a data processing task using a CSV or JSON input.",
]
```

For each seed, pick 1-2 random mutations. Call DeepSeek chat (cheap, non-reasoning
model) to generate the mutated prompt. This is a simple text-to-text call, no
tool use needed.

Cost estimate: 100K prompts × ~200 tokens each = 20M tokens ≈ $2-5.

---

## 3. Agent Harness

A Python script that:
1. Takes a prompt
2. Sends it to DeepSeek V3.2 with tool definitions
3. When the model returns a tool call, executes it in a sandbox
4. Feeds the result back to the model
5. Repeats until the model gives a final answer
6. Records the full trace

### DeepSeek API setup

DeepSeek's API is OpenAI-compatible. Use the OpenAI Python SDK pointed at
DeepSeek's endpoint:

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com"
)
```

### Tool definitions

Start with one tool — Python execution:

```python
TOOLS = [{
    "type": "function",
    "function": {
        "name": "python",
        "description": "Execute Python code and return stdout. Use this to compute answers, test solutions, or verify results.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
}]
```

### Agent loop

```python
def run_agent(prompt, max_turns=5):
    """Run one agent trace. Returns the full conversation history."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    for turn in range(max_turns):
        response = client.chat.completions.create(
            model="deepseek-chat",  # V3.2
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)

        # if model made tool calls, execute them
        if msg.tool_calls:
            for tc in msg.tool_calls:
                result = execute_python(tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            break  # model gave final text response

    return messages
```

### System prompt

```
You are a coding assistant. Solve the user's task step by step.
Use the python tool to write and run code. Always verify your
answer by running it. Give your final answer after verification.
```

### Python sandbox

Execute code in a subprocess with timeout and resource limits:

```python
import subprocess

def execute_python(code_json):
    """Execute Python code in sandbox, return stdout or error."""
    code = json.loads(code_json)["code"]
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout[:2000]  # truncate long outputs
        return f"Error: {result.stderr[:1000]}"
    except subprocess.TimeoutExpired:
        return "Error: execution timed out (10s)"
```

---

## 4. Trace Generation Pipeline

```python
def generate_traces(prompts, output_file, batch_size=50):
    """Generate agent traces for all prompts, save incrementally."""
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        for prompt in batch:
            trace = run_agent(prompt["text"])
            correct = verify_answer(trace, prompt["expected"])
            if correct:
                save_trace(output_file, trace, prompt)
        print(f"  {i+batch_size}/{len(prompts)} done")
```

### Verification

Each dataset has ground truth. Check the model's final answer against it:

- **APPS/MBPP**: Run the generated code against test cases
- **GSM8K**: Extract numeric answer, compare to ground truth
- **MATH**: Parse final answer, compare symbolically
- **HumanEval**: Run generated function against assertions

### Parallelization

Run 10-20 concurrent API calls (DeepSeek allows high rate limits).
Use asyncio or multiprocessing. 100K traces at ~3s each with 20 workers ≈ 4 hours.

### Cost estimate

```
100K traces × ~5 API calls each × ~500 tokens per call = 250M tokens
DeepSeek V3.2 pricing: ~$0.27/M input, ~$1.10/M output (as of 2026)
Estimated cost: ~$100-200

After filtering (keep ~50% correct): ~50K training traces
```

---

## 5. SFT Training Format

### Trace format for training

Convert each API conversation trace into our model's training format:

```
<|user|>
Write a function to find the longest palindromic substring.
<|assistant|>
<tool_call>python
def longest_palindrome(s):
    if not s:
        return ""
    best = s[0]
    for i in range(len(s)):
        for lo, hi in [(i, i), (i, i+1)]:
            while lo >= 0 and hi < len(s) and s[lo] == s[hi]:
                lo -= 1
                hi += 1
            if hi - lo - 1 > len(best):
                best = s[lo+1:hi]
    return best

# test
print(longest_palindrome("babad"))
print(longest_palindrome("cbbd"))
</tool_call>
<|tool_result|>
bab
bb
<|tool_result_end|>
The function works. `longest_palindrome` expands around each center position and tracks the longest palindrome found.
<|end|>
```

### Loss masking (critical)

During SFT, only compute loss on tokens the MODEL generates. Everything else is
context (input) that the model reads but is not trained to produce:

```
MASKED (no loss):     <|user|> ... <|assistant|>
TRAINED (compute loss): <tool_call>python\ndef longest_palindrome...
MASKED (no loss):     <|tool_result|> bab\nbb <|tool_result_end|>
TRAINED (compute loss): The function works...
MASKED (no loss):     <|end|>
```

Implementation in train.py: use a `loss_mask` array alongside targets. Set mask=0
for user turns, tool results, and special tokens. Set mask=1 for assistant outputs.

```python
# In the loss function, multiply by mask before averaging:
token_losses = -log_probs_at_targets  # (batch, seq)
masked_losses = token_losses * loss_mask  # zero out non-model tokens
loss = masked_losses.sum() / loss_mask.sum()  # average over model tokens only
```

### Special tokens

Add to tokenizer vocabulary (4 new tokens):

```
<|user|>          — start of user message
<|assistant|>     — start of model response
<|tool_result|>   — start of tool output (model doesn't generate this)
<|tool_result_end|> — end of tool output
```

Alternatively, use simple text markers that are already in the BPE vocabulary
to avoid modifying the tokenizer:

```
User:
Assistant:
[TOOL_CALL]python
[TOOL_RESULT]
[/TOOL_RESULT]
```

The second approach avoids retraining the embedding layer for new tokens — the
model already knows these words/characters from pretraining.

---

## 6. SFT Training

### Training setup

```bash
# After v3 pretraining completes:
uv run python -u train_sft.py \
    --resume weights_v3.pkl \
    --data traces_filtered.jsonl \
    --epochs 2 \
    --batch-size 16 \
    --lr 2e-5 \
    --warmup-steps 100
```

### Key parameters

- **Learning rate**: 2e-5 (much lower than pretraining — we're fine-tuning, not
  training from scratch. Too high and the model forgets pretraining.)
- **Epochs**: 1-2 (more causes overfitting to the SFT format)
- **Batch size**: 16 (same as pretraining on 4080 Super)
- **Context length**: 512 (same as pretraining, traces must fit)
- **Steps**: ~50K examples / 16 per batch = ~3,125 steps per epoch. Total ~6K steps.
- **Time**: ~1-2 hours on H200

### What to build

`train_sft.py` — a variant of `train.py` that:
1. Loads JSONL traces instead of raw token stream
2. Tokenizes each trace with role markers
3. Creates loss_mask (1 for assistant tokens, 0 for everything else)
4. Uses masked cross-entropy loss
5. No curriculum (traces are already varied length)

---

## 7. After SFT: RL with Execution Rewards

Once the model can produce tool-call-formatted output, apply GRPO:

1. Give model a coding prompt
2. Model generates a trace (possibly with tool calls)
3. Execute the code, run test cases
4. Reward = fraction of tests passed (0.0 to 1.0)
5. GRPO: sample N completions per prompt, rank by reward, update policy

This is the stage where the model actually gets BETTER at coding — SFT just
taught it the format, RL teaches it to be correct.

---

## 8. Summary and Dependencies

```
Step                  Input               Output              Cost     Time
─────────────────────────────────────────────────────────────────────────────
1. Download seeds     HuggingFace         36K prompts         free     5 min
2. Augment prompts    36K seeds           100K prompts        ~$5      1 hr
3. Generate traces    100K prompts        100K traces         ~$150    4 hrs
4. Filter traces      100K traces         ~50K verified       free     2 hrs
5. Format for SFT     50K traces          training JSONL      free     10 min
6. Train SFT          JSONL + weights     sft_weights.pkl     ~$5      2 hrs
7. RL (GRPO)          sft_weights + tasks rl_weights.pkl      ~$10     8 hrs
─────────────────────────────────────────────────────────────────────────────
Total                                                         ~$170    ~17 hrs
```

### Files to create

```
generate_sft_traces.py    — steps 1-5: download seeds, augment, run agent, filter, format
train_sft.py              — step 6: masked SFT training loop
train_rl.py               — step 7: GRPO training with execution rewards
```
