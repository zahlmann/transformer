# H100 Spot Instance Training — 1 Epoch

306M transformer, 7.85B tokens, single H100 80GB. Expected ~14-15h, ~$14 total.

## On the GPU server

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

# 5. Download and prepare training data (~1-2h)
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -u prepare_data_v2.py

# 6. Train (1 epoch, checkpoints every 5000 steps)
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

Resume from your backed-up checkpoint (see below):

```bash
scp ~/transformer-backup/checkpoint.pkl user@NEW_SERVER:~/transformer/

# Then on the new server, after setup steps 1-5:
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 64 --epochs 1 \
  --curriculum --lr 3e-4 --no-checkpoint \
  --checkpoint-interval 5000 \
  --resume checkpoint.pkl \
  2>&1 | tee -a training.log
```

### When training finishes

The final model is saved to `weights.pkl` (~1.2GB). Copy it home:

```bash
# On your local machine:
scp user@SERVER_IP:~/transformer/weights.pkl ~/transformer-backup/
```

---

## On your local machine — checkpoint backup

Run this in a terminal while training runs. Pulls checkpoint every 5 minutes.

```bash
#!/bin/bash
# backup_checkpoints.sh
# Usage: bash backup_checkpoints.sh user@server-ip

SERVER=${1:?Usage: bash backup_checkpoints.sh user@server-ip}
BACKUP_DIR=~/transformer-backup
mkdir -p "$BACKUP_DIR"

echo "Backing up checkpoints from $SERVER every 5 minutes..."
echo "Saving to $BACKUP_DIR"
echo "Press Ctrl+C to stop."

while true; do
  rsync -azP "$SERVER:~/transformer/checkpoint.pkl" "$BACKUP_DIR/" 2>/dev/null && \
    echo "[$(date +%H:%M)] checkpoint synced" || \
    echo "[$(date +%H:%M)] no checkpoint yet or server unreachable"
  rsync -azP "$SERVER:~/transformer/training.log" "$BACKUP_DIR/" 2>/dev/null
  sleep 300
done
```

Run it:

```bash
mkdir -p ~/transformer-backup
bash backup_checkpoints.sh user@SERVER_IP
```

Keep this terminal open for the duration of training.
