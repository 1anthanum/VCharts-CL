#!/bin/bash
# ============================================================
# Run all VTBench experiments (5A-5H) sequentially
# Designed for Lambda Cloud or any Linux GPU server
# ============================================================
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   nohup bash scripts/run_all_experiments.sh > experiment_all.log 2>&1 &
#
# Prerequisites:
#   pip install -e .
#   export CHART_IMAGE_ROOT=/path/to/chart_images  (if not default)
#
# ============================================================

set -e

LOGDIR="results/logs"
mkdir -p "$LOGDIR"

echo "============================================================"
echo "VTBench Full Experiment Suite"
echo "Started: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "============================================================"

# --- 5A: Training-Time Augmentation ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5A..."
python scripts/experiment_5a_train_augmentation.py \
    --config vtbench/config/experiment_5a_full.yaml \
    2>&1 | tee "$LOGDIR/exp5a.log"
echo "[$(date '+%H:%M:%S')] 5A complete."

# --- 5B: Multi-Chart Fusion ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5B..."
python scripts/experiment_5b_multi_chart.py \
    --config vtbench/config/experiment_5b_full.yaml \
    2>&1 | tee "$LOGDIR/exp5b.log"
echo "[$(date '+%H:%M:%S')] 5B complete."

# --- 5C: ResNet18 ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5C..."
python scripts/experiment_5c_resnet18.py \
    --config vtbench/config/experiment_5c_full.yaml \
    2>&1 | tee "$LOGDIR/exp5c.log"
echo "[$(date '+%H:%M:%S')] 5C complete."

# --- 5D: Rendering Optimization ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5D..."
python scripts/experiment_5d_rendering.py \
    --config vtbench/config/experiment_5d_full.yaml \
    2>&1 | tee "$LOGDIR/exp5d.log"
echo "[$(date '+%H:%M:%S')] 5D complete."

# --- 5E: Two-Branch Fusion ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5E..."
python scripts/experiment_5e_two_branch.py \
    --config vtbench/config/experiment_5e_full.yaml \
    2>&1 | tee "$LOGDIR/exp5e.log"
echo "[$(date '+%H:%M:%S')] 5E complete."

# --- 5F: Dataset Scale Effect ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5F..."
python scripts/experiment_5f_scale.py \
    --config vtbench/config/experiment_5f_scale.yaml \
    2>&1 | tee "$LOGDIR/exp5f.log"
echo "[$(date '+%H:%M:%S')] 5F complete."

# --- 5G: Chart Type Comparison ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5G..."
python scripts/experiment_5g_chart_type.py \
    --config vtbench/config/experiment_5g_chart_type.yaml \
    2>&1 | tee "$LOGDIR/exp5g.log"
echo "[$(date '+%H:%M:%S')] 5G complete."

# --- 5H: Image Resolution Effect ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5H..."
python scripts/experiment_5h_resolution.py \
    --config vtbench/config/experiment_5h_resolution.yaml \
    2>&1 | tee "$LOGDIR/exp5h.log"
echo "[$(date '+%H:%M:%S')] 5H complete."

# --- 5I: Modern Backbone Comparison ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5I..."
python scripts/experiment_5i_backbones.py \
    --config vtbench/config/experiment_5i_backbones.yaml \
    2>&1 | tee "$LOGDIR/exp5i.log"
echo "[$(date '+%H:%M:%S')] 5I complete."

# --- 5J: Training Strategy Optimization ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5J..."
python scripts/experiment_5j_training.py \
    --config vtbench/config/experiment_5j_training.yaml \
    2>&1 | tee "$LOGDIR/exp5j.log"
echo "[$(date '+%H:%M:%S')] 5J complete."

# --- 5K: Time-Series Augmentation ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5K..."
python scripts/experiment_5k_ts_augment.py \
    --config vtbench/config/experiment_5k_ts_augment.yaml \
    2>&1 | tee "$LOGDIR/exp5k.log"
echo "[$(date '+%H:%M:%S')] 5K complete."

# --- 5L: Ensemble Voting ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5L..."
python scripts/experiment_5l_ensemble.py \
    --config vtbench/config/experiment_5l_ensemble.yaml \
    2>&1 | tee "$LOGDIR/exp5l.log"
echo "[$(date '+%H:%M:%S')] 5L complete."

# --- 5M: Cross-Dataset Transfer Learning ---
echo ""
echo "[$(date '+%H:%M:%S')] Starting Experiment 5M..."
python scripts/experiment_5m_transfer.py \
    --config vtbench/config/experiment_5m_transfer.yaml \
    2>&1 | tee "$LOGDIR/exp5m.log"
echo "[$(date '+%H:%M:%S')] 5M complete."

echo ""
echo "============================================================"
echo "ALL EXPERIMENTS COMPLETE: $(date)"
echo "Results in: results/"
echo "Logs in:    $LOGDIR/"
echo "============================================================"
