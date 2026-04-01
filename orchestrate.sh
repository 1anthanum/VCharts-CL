#!/bin/bash
#SBATCH --job-name=vtbench-full
#SBATCH --partition=low
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/logs/%j_out.txt
#SBATCH --error=/home/%u/logs/%j_err.txt
#SBATCH --signal=USR1@300
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=ellanthanum@gmail.com

# ── Signal handler: upload partial results before Slurm kills us ──
cleanup() {
    echo "[$(date)] Signal received — uploading partial results to R2..."
    rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/" 2>/dev/null
    echo "[$(date)] Partial upload done. Job ending."
    exit 0
}
trap cleanup SIGTERM SIGINT USR1

# ── Pull latest code (fail-safe) ──
cd ~/myproject && git pull --ff-only || echo "WARN: git pull failed, using existing code"

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

# ── Activate environment ──
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench
export PATH=$HOME/bin:$PATH

# ── CRITICAL: Set image root to Linux path ──
export CHART_IMAGE_ROOT=/tmp/chart_images

# ── Print diagnostic info ──
echo "[$(date)] Job $SLURM_JOB_ID started on $(hostname)"
echo "CHART_IMAGE_ROOT=$CHART_IMAGE_ROOT"
echo "Python: $(which python) $(python --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARN: nvidia-smi failed"

# ── Run full experiment suite (--resume skips completed experiments) ──
python scripts/orchestrator.py --resume --solo

# ── Upload results to R2 ──
echo "[$(date)] Uploading results to R2..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Upload complete. Job finished."
