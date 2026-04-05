#!/bin/bash
#SBATCH --job-name=vtb-A-core
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/%u/logs/%j_out.txt
#SBATCH --error=/home/%u/logs/%j_err.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ellanthanum@gmail.com

# ════════════════════════════════════════════════════
# Job A: 6b (numerical baselines) + 6a (encoding comparison)
# Priority: HIGHEST — paper positioning + core comparison
# Estimated: 6b ~30min + 6a ~360min ≈ 7h
# ════════════════════════════════════════════════════

export PATH=$HOME/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

# Verify environment
python -c "import yaml; import torch; print(f'yaml OK, CUDA={torch.cuda.is_available()}')" || { echo "FATAL: env broken"; exit 1; }

cd ~/myproject

# ── Data setup ──
if [ ! -d /tmp/UCRArchive_2018 ]; then
    echo "[$(date)] Downloading UCR data..."
    rclone copy r2:ucr-experiments/UCRArchive_2018.zip /tmp/
    unzip -q -P "someone" /tmp/UCRArchive_2018.zip -d /tmp/
    rm -f /tmp/UCRArchive_2018.zip
fi
ln -sfn /tmp/UCRArchive_2018 ~/myproject/UCRArchive_2018

if [ ! -d /tmp/chart_images ] || [ -z "$(ls -A /tmp/chart_images 2>/dev/null)" ]; then
    echo "[$(date)] Downloading chart images..."
    rclone copy r2:ucr-experiments/chart_images/ /tmp/chart_images/ --progress
fi
export CHART_IMAGE_ROOT=/tmp/chart_images
export WANDB_DISABLED=true

echo "[$(date)] Job A started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── Run 6b first (fast, ~30 min) ──
echo "[$(date)] === Starting 6b: Numerical Baselines ==="
python scripts/experiment_6b_numerical_baseline.py --config vtbench/config/experiment_6b_numerical.yaml
echo "[$(date)] === 6b finished ==="

# ── Then 6a (core, ~6h) ──
echo "[$(date)] === Starting 6a: Encoding Comparison ==="
python scripts/experiment_6a_encodings.py --config vtbench/config/experiment_6a_encodings.yaml
echo "[$(date)] === 6a finished ==="

# ── Upload ──
echo "[$(date)] Uploading results..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job A complete."
