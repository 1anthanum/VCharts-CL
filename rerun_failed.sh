#!/bin/bash
#SBATCH --job-name=vtbench-rerun
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/%u/logs/%j_out.txt
#SBATCH --error=/home/%u/logs/%j_err.txt
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=ellanthanum@gmail.com

# ── CRITICAL: Set PATH early so rclone (in ~/bin) is available ──
export PATH=$HOME/bin:$PATH

# ── Activate environment ──
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

# ── Signal handler: upload partial results before Slurm kills us ──
cleanup() {
    echo "[$(date)] Signal received — uploading partial results to R2..."
    rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/" 2>/dev/null
    echo "[$(date)] Partial upload done. Job ending."
    exit 0
}
trap cleanup SIGTERM SIGINT USR1

cd ~/myproject

# ── Verify conda environment ──
echo "[$(date)] Python: $(which python) $(python --version 2>&1)"
python -c "import yaml; print('yaml OK')" || { echo "FATAL: yaml not found — conda env not activated"; exit 1; }
echo "[$(date)] Conda environment verified."

# ── Download & extract UCR data (skip if already present) ──
if [ ! -d /tmp/UCRArchive_2018 ]; then
    echo "[$(date)] Downloading UCR data from R2..."
    rclone copy r2:ucr-experiments/UCRArchive_2018.zip /tmp/
    unzip -q -P "someone" /tmp/UCRArchive_2018.zip -d /tmp/
    rm -f /tmp/UCRArchive_2018.zip
    echo "[$(date)] UCR data extracted."
else
    echo "[$(date)] UCR data already in /tmp/, skipping download."
fi
ln -sfn /tmp/UCRArchive_2018 ~/myproject/UCRArchive_2018

# ── Download pre-generated chart images (skip if already present) ──
if [ ! -d /tmp/chart_images ] || [ -z "$(ls -A /tmp/chart_images 2>/dev/null)" ]; then
    echo "[$(date)] Downloading chart images from R2..."
    rclone copy r2:ucr-experiments/chart_images/ /tmp/chart_images/ --progress
    echo "[$(date)] Chart images downloaded."
else
    echo "[$(date)] Chart images already in /tmp/, skipping download."
fi

# ── CRITICAL: Set image root to Linux path ──
export CHART_IMAGE_ROOT=/tmp/chart_images

# ── Disable W&B ──
export WANDB_DISABLED=true

# ── Print diagnostic info ──
echo "[$(date)] Job $SLURM_JOB_ID started on $(hostname)"
echo "CHART_IMAGE_ROOT=$CHART_IMAGE_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARN: nvidia-smi failed"

# ── Step 1: Fix orchestrator_state.json ──
# Remove falsely "completed" entries (6a/6b/8a/8b/9a/pregen_9a ran in 0.1 min = never actually ran)
echo "[$(date)] Fixing orchestrator_state.json — removing false completions..."
python -c "
import json
with open('results/orchestrator_state.json', 'r') as f:
    state = json.load(f)

# Remove falsely completed experiments (ran in 0.1 min on Apr 1 = no data)
false_completed = ['6a', '6b', '8a', '8b', '9a', 'pregen_9a', 'preflight']
removed = []
for key in false_completed:
    if key in state.get('completed', {}):
        del state['completed'][key]
        removed.append(key)

# Also remove 5g/5h/5j/5l from 'failed' so orchestrator will retry them
retry_failed = ['5g', '5h', '5j', '5l']
for key in retry_failed:
    if key in state.get('failed', {}):
        del state['failed'][key]
        removed.append(key + '(failed)')

print(f'Removed from state: {removed}')
print(f'Remaining completed: {list(state[\"completed\"].keys())}')

with open('results/orchestrator_state.json', 'w') as f:
    json.dump(state, f, indent=2)
print('State file updated successfully.')
"

# ── Step 2: Run the failed experiments ──
# --only targets specific experiments; orchestrator skips already-completed ones
echo "[$(date)] Starting experiment re-runs..."
python scripts/orchestrator.py --only 6b 6a 8a 8b 9a 5g 5h 5j 5l --solo

# ── Upload results to R2 ──
echo "[$(date)] Uploading results to R2..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Upload complete. Job finished."
