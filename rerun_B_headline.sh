#!/bin/bash
#SBATCH --job-name=vtb-B-8a
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/%u/logs/%j_out.txt
#SBATCH --error=/home/%u/logs/%j_err.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ellanthanum@gmail.com

# ════════════════════════════════════════════════════
# Job B: 8a (broad evaluation — HEADLINE experiment)
# Priority: HIGHEST — 20 datasets × 9 encodings × 5 seeds
# Estimated: ~10-14h
# ════════════════════════════════════════════════════

export PATH=$HOME/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

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

echo "[$(date)] Job B started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── Run 8a (headline experiment) ──
echo "[$(date)] === Starting 8a: Broad Dataset Evaluation ==="
python scripts/experiment_8a_broad_evaluation.py --config vtbench/config/experiment_8a_broad.yaml
echo "[$(date)] === 8a finished ==="

# ── Upload ──
echo "[$(date)] Uploading results..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job B complete."
