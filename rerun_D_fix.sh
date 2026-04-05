#!/bin/bash
#SBATCH --job-name=vtb-D-fix
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
# Job D: Re-run 8b + 9a with ROBUST data download
# Fix: verify actual data files, not just directory existence
# ════════════════════════════════════════════════════

export PATH=$HOME/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

python -c "import yaml; import torch; print(f'yaml OK, CUDA={torch.cuda.is_available()}')" || { echo "FATAL: env broken"; exit 1; }

cd ~/myproject

# ── ROBUST data check: verify actual dataset files, not just directory ──
PROBE_FILE="/tmp/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv"
if [ ! -f "$PROBE_FILE" ]; then
    echo "[$(date)] UCR data missing or incomplete — forcing fresh download..."
    rm -rf /tmp/UCRArchive_2018
    rclone copy r2:ucr-experiments/UCRArchive_2018.zip /tmp/
    unzip -q -P "someone" /tmp/UCRArchive_2018.zip -d /tmp/
    rm -f /tmp/UCRArchive_2018.zip
    echo "[$(date)] UCR data extracted."
    # Verify
    if [ ! -f "$PROBE_FILE" ]; then
        echo "FATAL: Download succeeded but probe file still missing!"
        ls -la /tmp/UCRArchive_2018/ 2>/dev/null | head -20
        exit 1
    fi
else
    echo "[$(date)] UCR data verified at $PROBE_FILE"
fi
ln -sfn /tmp/UCRArchive_2018 ~/myproject/UCRArchive_2018

# ── Verify symlink actually works ──
if [ ! -f "UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv" ]; then
    echo "FATAL: Symlink broken — UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv not accessible"
    ls -la ~/myproject/UCRArchive_2018 2>/dev/null
    exit 1
fi
echo "[$(date)] Symlink verified OK"

# ── Chart images ──
if [ ! -d /tmp/chart_images ] || [ -z "$(ls -A /tmp/chart_images 2>/dev/null)" ]; then
    echo "[$(date)] Downloading chart images..."
    rclone copy r2:ucr-experiments/chart_images/ /tmp/chart_images/ --progress
fi
export CHART_IMAGE_ROOT=/tmp/chart_images
export WANDB_DISABLED=true

echo "[$(date)] Job D started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ── Delete old empty results so scripts don't resume from nothing ──
echo "[$(date)] Clearing old empty results for 8b and 9a..."
rm -f results/experiment_8b_compute_profile/compute_profile.csv
rm -f results/experiment_9a_advanced/advanced_results.csv

# ── 8b: Compute profiling ──
echo "[$(date)] === Starting 8b: Compute Profiling ==="
python scripts/experiment_8b_compute_profile.py --config vtbench/config/experiment_8b_compute.yaml
echo "[$(date)] === 8b finished ==="

# ── 9a: Advanced methods ──
echo "[$(date)] === Starting 9a: Advanced Methods ==="
python scripts/experiment_9a_advanced.py --config vtbench/config/experiment_9a_advanced.yaml
echo "[$(date)] === 9a finished ==="

# ── Upload ──
echo "[$(date)] Uploading results..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job D complete."
