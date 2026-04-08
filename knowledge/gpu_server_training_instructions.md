# H100 Spot Instance Training — 1 Epoch

306M transformer, 7.85B tokens, single H100 80GB. Expected ~14-15h, ~$14 total.

## Setup and training

```bash
# 1. Install tools
apt-get update && apt-get install -y tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 2. Clone repo
git clone https://github.com/zahlmann/transformer.git
cd transformer

# 3. Install dependencies
uv sync

# 4. Start a tmux session (survives disconnects)
tmux new -s train

# 5. Set HuggingFace token (paste your read token here)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 6. Download and prepare training data (~1-2h)
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -u prepare_data_v2.py

# 7. Train (1 epoch, checkpoints every 5000 steps)
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 64 --epochs 1 \
  --curriculum --lr 3e-4 --no-checkpoint \
  --checkpoint-interval 5000 \
  2>&1 | tee training.log
```

### Batch size tuning

- Start with `--batch-size 64` (~35GB VRAM estimated)
- If it works, try 96 or 128 for potentially more speed
- If it OOMs, fall back to 48
- Curriculum multiplies batch size in early phases (bs×4 at ctx=128, bs×2 at ctx=256)

### If the spot instance dies

Storage persists across instances. On the new instance, skip data download and resume:

```bash
# 1. Install tools
apt-get update && apt-get install -y tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 2. Go to repo (already on persistent storage)
cd transformer

# 3. Install dependencies
uv sync

# 4. Resume training in tmux
tmux new -s train
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 64 --epochs 1 \
  --curriculum --lr 3e-4 --no-checkpoint \
  --checkpoint-interval 5000 \
  --resume checkpoint.pkl \
  2>&1 | tee -a training.log
```

### When training finishes

The final model is saved to `weights.pkl` (~1.2GB). Download it from persistent storage.
