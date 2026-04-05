#!/bin/bash
#SBATCH --job-name=vtb-D2-supp
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
# Job D v2: 8b + 9a + re-verify 5g/5h/5j/5l
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
echo "[$(date)] Job D v2 started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── Clear old empty results ──
rm -f results/experiment_8b_compute_profile/compute_profile.csv 2>/dev/null
rm -f results/experiment_9a_advanced/advanced_results.csv 2>/dev/null

# ── 8b ──
echo "[$(date)] === Starting 8b ==="
python scripts/experiment_8b_compute_profile.py --config vtbench/config/experiment_8b_compute.yaml
echo "[$(date)] === 8b finished ==="

# ── 9a ──
echo "[$(date)] === Starting 9a ==="
python scripts/experiment_9a_advanced.py --config vtbench/config/experiment_9a_advanced.yaml
echo "[$(date)] === 9a finished ==="

# ── Re-run 5g/5h/5j/5l to ensure completeness ──
echo "[$(date)] === Starting 5g (re-verify) ==="
python scripts/experiment_5g_chart_type.py --config vtbench/config/experiment_5g_chart_type.yaml
echo "[$(date)] === 5g finished ==="

echo "[$(date)] === Starting 5h (re-verify) ==="
python scripts/experiment_5h_resolution.py --config vtbench/config/experiment_5h_resolution.yaml
echo "[$(date)] === 5h finished ==="

echo "[$(date)] === Starting 5j (re-verify) ==="
python scripts/experiment_5j_training.py --config vtbench/config/experiment_5j_training.yaml
echo "[$(date)] === 5j finished ==="

echo "[$(date)] === Starting 5l (re-verify) ==="
python scripts/experiment_5l_ensemble.py --config vtbench/config/experiment_5l_ensemble.yaml
echo "[$(date)] === 5l finished ==="

rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job D v2 complete."
