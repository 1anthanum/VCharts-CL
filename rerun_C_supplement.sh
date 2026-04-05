#!/bin/bash
#SBATCH --job-name=vtb-C-supp
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
# Job C: 8b + 9a + 5g/5h/5j/5l (supplementary experiments)
# Priority: MEDIUM — compute profiling + ablation supplements
# Estimated: ~8-12h total
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

echo "[$(date)] Job C started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── 8b: Compute profiling (~60 min) ──
echo "[$(date)] === Starting 8b: Compute Profiling ==="
python scripts/experiment_8b_compute_profile.py --config vtbench/config/experiment_8b_compute.yaml
echo "[$(date)] === 8b finished ==="

# ── 9a: Advanced methods (~600 min) ──
echo "[$(date)] === Starting 9a: Advanced Methods ==="
python scripts/experiment_9a_advanced.py --config vtbench/config/experiment_9a_advanced.yaml
echo "[$(date)] === 9a finished ==="

# ── 5g: Chart type comparison (~45 min) ──
echo "[$(date)] === Starting 5g: Chart Type Comparison ==="
python scripts/experiment_5g_chart_type.py --config vtbench/config/experiment_5g_chart_type.yaml
echo "[$(date)] === 5g finished ==="

# ── 5h: Resolution effect (~45 min) ──
echo "[$(date)] === Starting 5h: Resolution Effect ==="
python scripts/experiment_5h_resolution.py --config vtbench/config/experiment_5h_resolution.yaml
echo "[$(date)] === 5h finished ==="

# ── 5j: Training strategies (~90 min) ──
echo "[$(date)] === Starting 5j: Training Strategies ==="
python scripts/experiment_5j_training.py --config vtbench/config/experiment_5j_training.yaml
echo "[$(date)] === 5j finished ==="

# ── 5l: Ensemble voting (~60 min) ──
echo "[$(date)] === Starting 5l: Ensemble Voting ==="
python scripts/experiment_5l_ensemble.py --config vtbench/config/experiment_5l_ensemble.yaml
echo "[$(date)] === 5l finished ==="

# ── Upload ──
echo "[$(date)] Uploading results..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job C complete."
