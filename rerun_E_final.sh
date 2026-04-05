#!/bin/bash
#SBATCH --job-name=vtb-E-final
#SBATCH --partition=high
#SBATCH --account=publicgrp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/%u/logs/%j_out.txt
#SBATCH --error=/home/%u/logs/%j_err.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ellanthanum@gmail.com

# ════════════════════════════════════════════════════
# Job E: FINAL FIX — 6a + 8a + 9a + 8b
# Key fix: extract tar.gz chart images after rclone download
# ════════════════════════════════════════════════════

export PATH=$HOME/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vtbench

python -c "import yaml; import torch; print(f'yaml OK, CUDA={torch.cuda.is_available()}')" || { echo "FATAL: env broken"; exit 1; }

cd ~/myproject

# ══════ 1. UCR RAW DATA ══════
UCR_PROBE="/tmp/UCRArchive_2018/GunPoint/GunPoint_TRAIN.tsv"
if [ ! -f "$UCR_PROBE" ]; then
    echo "[$(date)] UCR data missing — downloading..."
    rm -rf /tmp/UCRArchive_2018
    rclone copy r2:ucr-experiments/UCRArchive_2018.zip /tmp/
    unzip -q -P "someone" /tmp/UCRArchive_2018.zip -d /tmp/
    rm -f /tmp/UCRArchive_2018.zip
    [ -f "$UCR_PROBE" ] || { echo "FATAL: UCR download failed"; exit 1; }
fi
ln -sfn /tmp/UCRArchive_2018 ~/myproject/UCRArchive_2018
echo "[$(date)] UCR data OK"

# ══════ 2. CHART IMAGES — download + EXTRACT tar.gz ══════
echo "[$(date)] Setting up chart images..."
rclone copy r2:ucr-experiments/chart_images/ /tmp/chart_images/ --progress

# KEY FIX: extract any tar.gz archives found
echo "[$(date)] Extracting tar.gz archives..."
EXTRACTED=0
for tarball in /tmp/chart_images/*.tar.gz; do
    [ -f "$tarball" ] || continue
    echo "  Extracting: $(basename $tarball)..."
    tar xzf "$tarball" -C /tmp/chart_images/ && rm -f "$tarball"
    EXTRACTED=$((EXTRACTED + 1))
done
echo "[$(date)] Extracted $EXTRACTED tar.gz archives"

# Also check for .tar files (no gz)
for tarball in /tmp/chart_images/*.tar; do
    [ -f "$tarball" ] || continue
    echo "  Extracting: $(basename $tarball)..."
    tar xf "$tarball" -C /tmp/chart_images/ && rm -f "$tarball"
    EXTRACTED=$((EXTRACTED + 1))
done

export CHART_IMAGE_ROOT=/tmp/chart_images

# ══════ 3. VERIFY IMAGE DIRECTORIES EXIST ══════
echo "[$(date)] Verifying chart image directories..."
MISSING=0
for ds in GunPoint SyntheticControl ECG5000 CBF Trace FordA Wafer; do
    DIR="/tmp/chart_images/${ds}_images"
    if [ ! -d "$DIR" ]; then
        echo "  WARNING: Missing $DIR"
        MISSING=$((MISSING + 1))
    fi
done

# If images still missing after extraction, generate them via preflight
if [ $MISSING -gt 0 ]; then
    echo "[$(date)] $MISSING datasets missing images — running preflight to generate..."
    python scripts/preflight_check.py 2>&1 | tail -20
    echo "[$(date)] Preflight complete"
fi

# Final verification
python -c "
import os
root = os.environ['CHART_IMAGE_ROOT']
missing = []
for ds in ['GunPoint', 'SyntheticControl', 'ECG5000', 'CBF', 'Trace']:
    d = os.path.join(root, f'{ds}_images')
    if not os.path.isdir(d):
        missing.append(ds)
    else:
        # Check at least some content exists
        contents = os.listdir(d)
        if len(contents) == 0:
            missing.append(f'{ds}(empty)')
if missing:
    print(f'WARNING: Still missing: {missing}')
else:
    print('All critical image directories verified OK')
print(f'Total datasets in chart_images: {len([d for d in os.listdir(root) if d.endswith(\"_images\")])}')
" || echo "WARNING: Python verification had issues"

export WANDB_DISABLED=true
echo "[$(date)] Job E started on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# ══════ 4. CLEAR OLD EMPTY RESULTS ══════
rm -f results/experiment_6a_encodings/encoding_comparison.csv
rm -f results/experiment_8a_broad_evaluation/broad_evaluation.csv
rm -f results/experiment_8b_compute_profile/compute_profile.csv
rm -f results/experiment_9a_advanced/advanced_results.csv

# ══════ 5. RUN EXPERIMENTS ══════

echo "[$(date)] === Starting 6a: Encoding Comparison ==="
python scripts/experiment_6a_encodings.py --config vtbench/config/experiment_6a_encodings.yaml
echo "[$(date)] === 6a finished ==="

echo "[$(date)] === Starting 8a: Broad Evaluation ==="
python scripts/experiment_8a_broad_evaluation.py --config vtbench/config/experiment_8a_broad.yaml
echo "[$(date)] === 8a finished ==="

echo "[$(date)] === Starting 8b: Compute Profiling ==="
python scripts/experiment_8b_compute_profile.py --config vtbench/config/experiment_8b_compute.yaml
echo "[$(date)] === 8b finished ==="

echo "[$(date)] === Starting 9a: Advanced Methods ==="
python scripts/experiment_9a_advanced.py --config vtbench/config/experiment_9a_advanced.yaml
echo "[$(date)] === 9a finished ==="

# ══════ 6. UPLOAD ══════
echo "[$(date)] Uploading results..."
rclone copy ~/myproject/results/ "r2:ucr-experiments/runs/${SLURM_JOB_ID}/"
echo "[$(date)] Job E complete."
