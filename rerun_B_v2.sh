#!/bin/bash
#SBATCH --job-name=vtb-B2-8a
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
# Job B v2: 8a (headline broad evaluation)
# FIXED: robust file-probe checks
# ════════════════════════════════════════════════════

export PATH=$HOME/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

python -c "import yaml; import torch; print(f'yaml OK, CUDA={torch.cuda.is_available()}')" || { echo "FATAL: env broken"; exit 1; }

cd ~/myproject

# ══════ ROBUST DATA SETUP ══════
UCR_PROBE="/tmp/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv"
if [ ! -f "$UCR_PROBE" ]; then
    echo "[$(date)] UCR data missing — forcing fresh download..."
    rm -rf /tmp/UCRArchive_2018
    rclone copy r2:ucr-experiments/UCRArchive_2018.zip /tmp/
    unzip -q -P "someone" /tmp/UCRArchive_2018.zip -d /tmp/
    rm -f /tmp/UCRArchive_2018.zip
    [ -f "$UCR_PROBE" ] || { echo "FATAL: UCR download failed"; exit 1; }
fi
ln -sfn /tmp/UCRArchive_2018 ~/myproject/UCRArchive_2018

IMG_PROBE="/tmp/chart_images/GunPoint_images/line_charts_color_with_label/train"
if [ ! -d "$IMG_PROBE" ]; then
    echo "[$(date)] Chart images missing — forcing fresh download..."
    rm -rf /tmp/chart_images
    rclone copy r2:ucr-experiments/chart_images/ /tmp/chart_images/ --progress
    [ -d "$IMG_PROBE" ] || { echo "FATAL: Chart images download failed"; exit 1; }
fi
export CHART_IMAGE_ROOT=/tmp/chart_images

python -c "
import os
assert os.path.isfile('UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv'), 'UCR MISSING'
assert os.path.isdir(os.environ['CHART_IMAGE_ROOT'] + '/GunPoint_images/line_charts_color_with_label/train'), 'IMG MISSING'
print('All data verified OK')
" || { echo "FATAL: verification failed"; exit 1; }

export WANDB_DISABLED=true
echo "[$(date)] Job B v2 started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── Clear old empty results ──
rm -f results/experiment_8a_broad/accuracy_results.csv 2>/dev/null

# ── 8a: Broad evaluation ──
echo "[$(date)] === Starting 8a ==="
python scripts/experiment_8a_broad_evaluation.py --config vtbench/config/experiment_8a_broad.yaml
echo "[$(date)] === 8a finished ==="

rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job B v2 complete."
