#!/bin/bash
# ============================================================
# Sync VTBench Experiment 5 files to SSH PC
# ============================================================
# Usage:
#   bash sync_to_ssh.sh user@host:/path/to/vtbench
#
# Example:
#   bash sync_to_ssh.sh ja@192.168.1.100:~/vtbench
#   bash sync_to_ssh.sh ja@mypc:~/projects/vtbench
# ============================================================

set -e

if [ -z "$1" ]; then
    echo "Usage: bash sync_to_ssh.sh user@host:/path/to/vtbench"
    echo ""
    echo "Example:"
    echo "  bash sync_to_ssh.sh ja@192.168.1.100:~/vtbench"
    exit 1
fi

TARGET="$1"
echo "=========================================="
echo "  Syncing VTBench Exp5 → ${TARGET}"
echo "=========================================="

# --- 1. Core bug fixes (trainer.py is critical) ---
echo ""
echo "[1/5] Core files (bug fixes)..."
rsync -avz --progress \
    vtbench/train/trainer.py \
    vtbench/train/evaluate.py \
    vtbench/data/loader.py \
    "${TARGET}/vtbench/train/" 2>/dev/null || \
rsync -avz --progress \
    vtbench/train/trainer.py \
    "${TARGET}/vtbench/train/"
rsync -avz --progress \
    vtbench/train/evaluate.py \
    "${TARGET}/vtbench/train/"
rsync -avz --progress \
    vtbench/data/loader.py \
    "${TARGET}/vtbench/data/"

# --- 2. Experiment 5 scripts ---
echo ""
echo "[2/5] Experiment scripts..."
rsync -avz --progress \
    scripts/experiment_5a_train_augmentation.py \
    scripts/experiment_5b_multi_chart.py \
    scripts/experiment_5c_resnet18.py \
    scripts/experiment_5d_rendering.py \
    scripts/experiment_5e_two_branch.py \
    "${TARGET}/scripts/"

# --- 3. Experiment 4 scripts (dependencies for 5A/5B/5C/5D/5E) ---
echo ""
echo "[3/5] Experiment 4 scripts (imported by Exp5)..."
rsync -avz --progress \
    scripts/experiment_4_augmentation_robustness.py \
    scripts/experiment_4_multiseed.py \
    "${TARGET}/scripts/"

# --- 4. Config files ---
echo ""
echo "[4/5] Config files..."
rsync -avz --progress \
    vtbench/config/experiment_5a_full.yaml \
    vtbench/config/experiment_5b_full.yaml \
    vtbench/config/experiment_5c_full.yaml \
    vtbench/config/experiment_5d_full.yaml \
    vtbench/config/experiment_5e_full.yaml \
    vtbench/config/experiment_5a_local.yaml \
    vtbench/config/experiment_5b_local.yaml \
    vtbench/config/experiment_5c_local.yaml \
    vtbench/config/experiment_5d_local.yaml \
    vtbench/config/experiment_5e_local.yaml \
    "${TARGET}/vtbench/config/"

# --- 5. Documentation ---
echo ""
echo "[5/5] Documentation..."
rsync -avz --progress \
    EXPERIMENT_5_README.md \
    "${TARGET}/"

echo ""
echo "=========================================="
echo "  Sync complete!"
echo "=========================================="
echo ""
echo "Next steps on SSH PC:"
echo "  cd $(echo ${TARGET} | cut -d: -f2)"
echo ""
echo "  # Quick smoke test (1 dataset, ~2 min):"
echo "  python scripts/experiment_5a_train_augmentation.py --config vtbench/config/experiment_5a_local.yaml"
echo ""
echo "  # Full run (5 datasets × 3 seeds):"
echo "  python scripts/experiment_5a_train_augmentation.py --config vtbench/config/experiment_5a_full.yaml"
echo "  python scripts/experiment_5b_multi_chart.py        --config vtbench/config/experiment_5b_full.yaml"
echo "  python scripts/experiment_5c_resnet18.py           --config vtbench/config/experiment_5c_full.yaml"
echo "  python scripts/experiment_5d_rendering.py          --config vtbench/config/experiment_5d_full.yaml"
echo "  python scripts/experiment_5e_two_branch.py         --config vtbench/config/experiment_5e_full.yaml"
