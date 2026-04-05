#!/bin/bash
#SBATCH --job-name=vtb-A2-core
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
# Job A v2: 6b + 6a (core experiments)
# FIXED: robust file-probe checks for both UCR data and chart images
# ════════════════════════════════════════════════════

export PATH=$HOME/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

python -c "import yaml; import torch; print(f'yaml OK, CUDA={torch.cuda.is_available()}')" || { echo "FATAL: env broken"; exit 1; }

cd ~/myproject

# ══════ ROBUST DATA SETUP ══════

# 1. UCR raw data: probe an actual dataset file
UCR_PROBE="/tmp/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv"
if [ ! -f "$UCR_PROBE" ]; then
    echo "[$(date)] UCR data missing/incomplete — forcing fresh download..."
    rm -rf /tmp/UCRArchive_2018
    rclone copy r2:ucr-experiments/UCRArchive_2018.zip /tmp/
    unzip -q -P "someone" /tmp/UCRArchive_2018.zip -d /tmp/
    rm -f /tmp/UCRArchive_2018.zip
    if [ ! -f "$UCR_PROBE" ]; then
        echo "FATAL: UCR download failed — probe file still missing"
        exit 1
    fi
    echo "[$(date)] UCR data verified."
else
    echo "[$(date)] UCR data OK at $UCR_PROBE"
fi
ln -sfn /tmp/UCRArchive_2018 ~/myproject/UCRArchive_2018

# 2. Chart images: probe an actual image directory
IMG_PROBE="/tmp/chart_images/GunPoint_images/line_charts_color_with_label/train"
if [ ! -d "$IMG_PROBE" ]; then
    echo "[$(date)] Chart images missing/incomplete — forcing fresh download..."
    rm -rf /tmp/chart_images
    rclone copy r2:ucr-experiments/chart_images/ /tmp/chart_images/ --progress
    if [ ! -d "$IMG_PROBE" ]; then
        echo "FATAL: Chart images download failed — probe dir still missing"
        exit 1
    fi
    echo "[$(date)] Chart images verified."
else
    echo "[$(date)] Chart images OK at $IMG_PROBE"
fi
export CHART_IMAGE_ROOT=/tmp/chart_images

# 3. Final verification
echo "[$(date)] Running Python data verification..."
python -c "
import os
ucr = 'UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv'
img = os.environ['CHART_IMAGE_ROOT'] + '/GunPoint_images/line_charts_color_with_label/train'
assert os.path.isfile(ucr), f'UCR MISSING: {ucr}'
assert os.path.isdir(img), f'IMG MISSING: {img}'
print('All data verified OK')
" || { echo "FATAL: Python verification failed"; exit 1; }

export WANDB_DISABLED=true
echo "[$(date)] Job A v2 started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── 6b: Numerical baselines (~30 min) ──
echo "[$(date)] === Starting 6b ==="
python scripts/experiment_6b_numerical_baseline.py --config vtbench/config/experiment_6b_numerical.yaml
echo "[$(date)] === 6b finished ==="

# ── 6a: Encoding comparison (~6h) ──
echo "[$(date)] === Starting 6a ==="
python scripts/experiment_6a_encodings.py --config vtbench/config/experiment_6a_encodings.yaml
echo "[$(date)] === 6a finished ==="

# ── Upload ──
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job A v2 complete."
