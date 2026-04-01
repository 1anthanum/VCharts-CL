# ============================================================
# VTBench Local 16-Hour Execution Plan (RTX 3070 Ti 8GB)
# ============================================================
# Strategy: Run 2 experiments in parallel (2-3 GB each)
# Total estimated: ~14-16 hours with parallelization
#
# PHASE 1 (0-1h):    5D completion + 5E start
# PHASE 2 (1-5h):    5E + 5I (backbone comparison - HIGHEST PRIORITY)
# PHASE 3 (5-9h):    5J (training) + 5K (TS augment)
# PHASE 4 (9-13h):   5H (resolution) + 5M (transfer)
# PHASE 5 (13-16h):  5L (ensemble) - needs all individual models first
#
# NOTE: 5F (scale, 216 runs) is deferred to A100 due to size
# ============================================================

$ErrorActionPreference = "Continue"
$LOGDIR = "results"
$env:CHART_IMAGE_ROOT = "D:\chart_images"

# Timestamp helper
function Log-Time { "[$(Get-Date -Format 'HH:mm:ss')]" }

Write-Host "$(Log-Time) === VTBench 16-Hour Local Run Starting ===" -ForegroundColor Cyan
Write-Host "$(Log-Time) GPU: RTX 3070 Ti 8GB" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# PHASE 1: Complete 5D (if needed) — ~20 min solo
# ============================================================
Write-Host "$(Log-Time) === PHASE 1: Completing 5D (Rendering) ===" -ForegroundColor Green

# Check if 5D results exist
if (-not (Test-Path "results/experiment_5d_full/accuracy_deepcnn.csv")) {
    Write-Host "$(Log-Time) 5D results not found, re-running..."
    python scripts/experiment_5d_rendering.py `
        --config vtbench/config/experiment_5d_full.yaml `
        2>&1 | Tee-Object -FilePath "$LOGDIR/exp5d_rerun.log"
    Write-Host "$(Log-Time) 5D complete."
} else {
    Write-Host "$(Log-Time) 5D results already exist, skipping."
}

# ============================================================
# PHASE 2: 5I (Backbones) + 5E (Two-Branch) in parallel
# ~7h for 5I, ~3.5h for 5E
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === PHASE 2: 5I (Backbones) + 5E (Two-Branch) parallel ===" -ForegroundColor Green

# Start 5I in background (HIGHEST PRIORITY experiment)
$job_5i = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    $env:CHART_IMAGE_ROOT = "D:\chart_images"
    conda activate vtbench
    python scripts/experiment_5i_backbones.py `
        --config vtbench/config/experiment_5i_backbones.yaml `
        2>&1 | Tee-Object -FilePath "results/exp5i.log"
}
Write-Host "$(Log-Time) 5I started (PID: $($job_5i.Id))"

# Start 5E in foreground
Write-Host "$(Log-Time) Starting 5E (Two-Branch)..."
python scripts/experiment_5e_two_branch.py `
    --config vtbench/config/experiment_5e_full.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5e.log"
Write-Host "$(Log-Time) 5E complete."

# Wait for 5I if still running
Write-Host "$(Log-Time) Waiting for 5I to finish..."
Wait-Job $job_5i
Receive-Job $job_5i
Write-Host "$(Log-Time) 5I complete."

# ============================================================
# PHASE 3: 5J (Training) + 5K (TS Augment) in parallel
# ~10.5h for 5J, ~6h for 5K
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === PHASE 3: 5J (Training) + 5K (TS Augment) parallel ===" -ForegroundColor Green

# Start 5K in background
$job_5k = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    $env:CHART_IMAGE_ROOT = "D:\chart_images"
    conda activate vtbench
    python scripts/experiment_5k_ts_augment.py `
        --config vtbench/config/experiment_5k_ts_augment.yaml `
        2>&1 | Tee-Object -FilePath "results/exp5k.log"
}
Write-Host "$(Log-Time) 5K started (PID: $($job_5k.Id))"

# 5J in foreground
Write-Host "$(Log-Time) Starting 5J (Training Enhancements)..."
python scripts/experiment_5j_training.py `
    --config vtbench/config/experiment_5j_training.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5j.log"
Write-Host "$(Log-Time) 5J complete."

Wait-Job $job_5k
Receive-Job $job_5k
Write-Host "$(Log-Time) 5K complete."

# ============================================================
# PHASE 4: 5H (Resolution) + 5M (Transfer) in parallel
# ~6h for 5H, ~4h for 5M
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === PHASE 4: 5H (Resolution) + 5M (Transfer) parallel ===" -ForegroundColor Green

$job_5m = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    $env:CHART_IMAGE_ROOT = "D:\chart_images"
    conda activate vtbench
    python scripts/experiment_5m_transfer.py `
        --config vtbench/config/experiment_5m_transfer.yaml `
        2>&1 | Tee-Object -FilePath "results/exp5m.log"
}
Write-Host "$(Log-Time) 5M started (PID: $($job_5m.Id))"

Write-Host "$(Log-Time) Starting 5H (Resolution)..."
python scripts/experiment_5h_resolution.py `
    --config vtbench/config/experiment_5h_resolution.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5h.log"
Write-Host "$(Log-Time) 5H complete."

Wait-Job $job_5m
Receive-Job $job_5m
Write-Host "$(Log-Time) 5M complete."

# ============================================================
# PHASE 5: 5L (Ensemble) — needs individual model results
# ~8h solo, last because it depends on trained models
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === PHASE 5: 5L (Ensemble) ===" -ForegroundColor Green
Write-Host "$(Log-Time) Starting 5L (Ensemble Voting)..."
python scripts/experiment_5l_ensemble.py `
    --config vtbench/config/experiment_5l_ensemble.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5l.log"
Write-Host "$(Log-Time) 5L complete."

# ============================================================
# DONE
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) ============================================" -ForegroundColor Cyan
Write-Host "$(Log-Time) ALL LOCAL EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host "$(Log-Time) ============================================" -ForegroundColor Cyan
Write-Host "$(Log-Time) Results saved to: results/"
Write-Host ""
Write-Host "Remaining for A100:"
Write-Host "  - 5F (Scale): 216 runs, ~10h"
Write-Host "  - ViT experiments"
Write-Host "  - Full 32-dataset benchmark"
