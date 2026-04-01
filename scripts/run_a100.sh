#!/bin/bash
# ============================================================
# VTBench A100 Execution Plan
# ============================================================
# A100 has 40GB+ VRAM — can run 4-6 experiments simultaneously
# Focus on: heavy experiments + ViT + full 32-dataset benchmark
#
# Estimated total: ~20-25h with heavy parallelization
# A100 effective throughput: 4x-6x of 3070 Ti
# ============================================================

set -e
LOGDIR="results"
mkdir -p "$LOGDIR"

timestamp() { date '+%H:%M:%S'; }

echo "[$(timestamp)] === VTBench A100 Run Starting ==="
echo "[$(timestamp)] GPU: NVIDIA A100"
echo ""

# ============================================================
# STEP 0: Environment setup (run lambda_setup.sh first)
# ============================================================
# source scripts/lambda_setup.sh

# ============================================================
# STEP 1: Image generation for all 32 datasets (if not uploaded)
# ============================================================
echo "[$(timestamp)] === STEP 1: Check/Generate images ==="
# If images were uploaded to R2 and downloaded:
# aws s3 sync s3://vtbench-images/chart_images/ chart_images/ --endpoint-url $R2_ENDPOINT

# Otherwise generate locally (A100 CPU is fast):
# python scripts/pregenerate_all_images.py

# ============================================================
# STEP 2: Run heavy experiments in parallel using GNU parallel
# ============================================================
echo "[$(timestamp)] === STEP 2: Parallel experiments ==="

# --- 5F (Scale): 216 runs, deferred from local ---
echo "[$(timestamp)] Starting 5F (Scale)..."
python scripts/experiment_5f_scale.py \
    --config vtbench/config/experiment_5f_scale.yaml \
    2>&1 | tee "$LOGDIR/exp5f.log" &
PID_5F=$!

# --- 5I with ViT enabled ---
echo "[$(timestamp)] Starting 5I+ViT (Backbones)..."
# Uncomment ViT in config first:
# sed -i 's/# - { name: "vit_tiny"/- { name: "vit_tiny"/' vtbench/config/experiment_5i_backbones.yaml
python scripts/experiment_5i_backbones.py \
    --config vtbench/config/experiment_5i_backbones.yaml \
    2>&1 | tee "$LOGDIR/exp5i_a100.log" &
PID_5I=$!

# --- 5H (Resolution) with higher batch size ---
echo "[$(timestamp)] Starting 5H (Resolution)..."
python scripts/experiment_5h_resolution.py \
    --config vtbench/config/experiment_5h_resolution.yaml \
    2>&1 | tee "$LOGDIR/exp5h.log" &
PID_5H=$!

# --- 5L (Ensemble) ---
echo "[$(timestamp)] Starting 5L (Ensemble)..."
python scripts/experiment_5l_ensemble.py \
    --config vtbench/config/experiment_5l_ensemble.yaml \
    2>&1 | tee "$LOGDIR/exp5l.log" &
PID_5L=$!

echo "[$(timestamp)] 4 experiments running in parallel (PIDs: $PID_5F $PID_5I $PID_5H $PID_5L)"

# Wait for all
wait $PID_5F && echo "[$(timestamp)] 5F complete." || echo "[$(timestamp)] 5F FAILED!"
wait $PID_5I && echo "[$(timestamp)] 5I+ViT complete." || echo "[$(timestamp)] 5I+ViT FAILED!"
wait $PID_5H && echo "[$(timestamp)] 5H complete." || echo "[$(timestamp)] 5H FAILED!"
wait $PID_5L && echo "[$(timestamp)] 5L complete." || echo "[$(timestamp)] 5L FAILED!"

# ============================================================
# STEP 3: Transfer learning + any remaining
# ============================================================
echo ""
echo "[$(timestamp)] === STEP 3: Transfer + remaining ==="

python scripts/experiment_5m_transfer.py \
    --config vtbench/config/experiment_5m_transfer.yaml \
    2>&1 | tee "$LOGDIR/exp5m.log" &

python scripts/experiment_5j_training.py \
    --config vtbench/config/experiment_5j_training.yaml \
    2>&1 | tee "$LOGDIR/exp5j.log" &

python scripts/experiment_5k_ts_augment.py \
    --config vtbench/config/experiment_5k_ts_augment.yaml \
    2>&1 | tee "$LOGDIR/exp5k.log" &

wait
echo "[$(timestamp)] All Step 3 experiments complete."

# ============================================================
# STEP 4: Full 32-dataset benchmark (if time allows)
# ============================================================
# echo "[$(timestamp)] === STEP 4: Full benchmark ==="
# python scripts/experiment_full_benchmark.py \
#     --config vtbench/config/experiment_full_benchmark.yaml \
#     2>&1 | tee "$LOGDIR/exp_full_benchmark.log"

echo ""
echo "[$(timestamp)] ============================================"
echo "[$(timestamp)] ALL A100 EXPERIMENTS COMPLETE!"
echo "[$(timestamp)] ============================================"
echo "[$(timestamp)] Uploading results..."

# Upload results back to R2
# aws s3 sync results/ s3://vtbench-results/ --endpoint-url $R2_ENDPOINT
