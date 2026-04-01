# ============================================================
# VTBench 16-Hour Combined Plan: Image Gen + Training
# ============================================================
# Phase A (0-3h):  Generate images for core datasets (CPU)
# Phase B (3-16h): Training (GPU) + background image gen (CPU)
#
# Strategy: Core dataset images FIRST, then start training
# immediately while extended/full datasets generate in background.
# ============================================================

$ErrorActionPreference = "Continue"
$env:CHART_IMAGE_ROOT = "D:\chart_images"
$LOGDIR = "results"

function Log-Time { "[$(Get-Date -Format 'HH:mm:ss')]" }

Write-Host "$(Log-Time) ================================================" -ForegroundColor Cyan
Write-Host "$(Log-Time)  VTBench 16-Hour Combined Execution" -ForegroundColor Cyan
Write-Host "$(Log-Time)  Phase A: Image Generation (core datasets)" -ForegroundColor Cyan
Write-Host "$(Log-Time)  Phase B: Training + background image gen" -ForegroundColor Cyan
Write-Host "$(Log-Time) ================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# PHASE A: Generate ALL encoding images for core 5 datasets
# Estimated: ~30-60 min (CPU only, no GPU needed)
# ============================================================
Write-Host "$(Log-Time) === PHASE A: Core Image Generation ===" -ForegroundColor Green

Write-Host "$(Log-Time) A1: Mathematical encodings (GASF/GADF/MTF/RP/CWT/STFT)..."
python scripts/pregenerate_all_encodings.py `
    --datasets SyntheticControl GunPoint CBF Trace ECG5000 FordA Wafer `
    --encodings gasf gadf mtf mtf_16 rp rp_grayscale cwt stft `
    --rgb-presets gasf_gadf_mtf rp_cwt_stft `
    --colormaps viridis plasma `
    --image-size 128
Write-Host "$(Log-Time) A1 complete."

Write-Host "$(Log-Time) A2: Chart images for experiment datasets..."
python scripts/pregenerate_all_images.py `
    --datasets SyntheticControl GunPoint CBF Trace ECG5000 FordA Wafer `
        BeetleFly TwoLeadECG Coffee Lightning2 ECG200 Computers Yoga `
        TwoPatterns `
    --chart-types line area bar scatter `
    --label-modes with_label
Write-Host "$(Log-Time) A2 complete."

Write-Host ""
Write-Host "$(Log-Time) === PHASE A DONE. Core images ready. ===" -ForegroundColor Green
Write-Host "$(Log-Time) Starting Phase B: Training + background gen..." -ForegroundColor Green
Write-Host ""

# ============================================================
# PHASE B: Start background image gen + GPU training
# ============================================================

# --- Background: Generate images for ALL 32 datasets ---
Write-Host "$(Log-Time) Starting background image generation for all 32 datasets..."
$bgJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    $env:CHART_IMAGE_ROOT = "D:\chart_images"

    # Extended + full datasets (skip core, already done)
    python scripts/pregenerate_all_encodings.py `
        --all-datasets `
        --encodings gasf gadf mtf rp rp_grayscale cwt stft `
        --rgb-presets gasf_gadf_mtf `
        --no-colormaps `
        --image-size 128

    # 10x TS augmentation images
    python scripts/pregenerate_5k_images.py --multipliers 10
}
Write-Host "$(Log-Time) Background image gen started (Job: $($bgJob.Id))"

# --- GPU Training Phase ---

# B1: 6B Numerical Baselines (~1h, lightweight, good warmup)
Write-Host ""
Write-Host "$(Log-Time) === B1: 6B Numerical Baselines ===" -ForegroundColor Yellow
python scripts/experiment_6b_numerical_baseline.py `
    --config vtbench/config/experiment_6b_numerical.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp6b.log"
Write-Host "$(Log-Time) 6B complete."

# B2: 5I Backbones (highest priority from original plan)
Write-Host ""
Write-Host "$(Log-Time) === B2: 5I Backbone Comparison ===" -ForegroundColor Yellow
python scripts/experiment_5i_backbones.py `
    --config vtbench/config/experiment_5i_backbones.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5i.log"
Write-Host "$(Log-Time) 5I complete."

# B3: 6A Encoding Comparison (THE key experiment)
Write-Host ""
Write-Host "$(Log-Time) === B3: 6A Encoding Comparison ===" -ForegroundColor Yellow
python scripts/experiment_6a_encodings.py `
    --config vtbench/config/experiment_6a_encodings.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp6a.log"
Write-Host "$(Log-Time) 6A complete."

# B4: 5E Two-Branch
Write-Host ""
Write-Host "$(Log-Time) === B4: 5E Two-Branch ===" -ForegroundColor Yellow
python scripts/experiment_5e_two_branch.py `
    --config vtbench/config/experiment_5e_full.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5e.log"
Write-Host "$(Log-Time) 5E complete."

# B5: 5J Training Enhancements
Write-Host ""
Write-Host "$(Log-Time) === B5: 5J Training Enhancements ===" -ForegroundColor Yellow
python scripts/experiment_5j_training.py `
    --config vtbench/config/experiment_5j_training.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5j.log"
Write-Host "$(Log-Time) 5J complete."

# B6: 5K TS Augmentation
Write-Host ""
Write-Host "$(Log-Time) === B6: 5K TS Augmentation ===" -ForegroundColor Yellow
python scripts/experiment_5k_ts_augment.py `
    --config vtbench/config/experiment_5k_ts_augment.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5k.log"
Write-Host "$(Log-Time) 5K complete."

# B7: 5H Resolution
Write-Host ""
Write-Host "$(Log-Time) === B7: 5H Resolution ===" -ForegroundColor Yellow
python scripts/experiment_5h_resolution.py `
    --config vtbench/config/experiment_5h_resolution.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5h.log"
Write-Host "$(Log-Time) 5H complete."

# B8: 5M Transfer Learning
Write-Host ""
Write-Host "$(Log-Time) === B8: 5M Transfer Learning ===" -ForegroundColor Yellow
python scripts/experiment_5m_transfer.py `
    --config vtbench/config/experiment_5m_transfer.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5m.log"
Write-Host "$(Log-Time) 5M complete."

# B9: 5L Ensemble (last — needs individual models)
Write-Host ""
Write-Host "$(Log-Time) === B9: 5L Ensemble ===" -ForegroundColor Yellow
python scripts/experiment_5l_ensemble.py `
    --config vtbench/config/experiment_5l_ensemble.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5l.log"
Write-Host "$(Log-Time) 5L complete."

# ============================================================
# Wait for background image gen
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) Checking background image generation..." -ForegroundColor Cyan
if ($bgJob.State -eq "Running") {
    Write-Host "$(Log-Time) Background gen still running. Waiting..."
    Wait-Job $bgJob
}
Receive-Job $bgJob -ErrorAction SilentlyContinue
Write-Host "$(Log-Time) Background image generation complete."

# ============================================================
# DONE
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) ================================================" -ForegroundColor Cyan
Write-Host "$(Log-Time)  ALL TASKS COMPLETE!" -ForegroundColor Cyan
Write-Host "$(Log-Time) ================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Completed experiments:"
Write-Host "  6B - Numerical Baselines"
Write-Host "  5I - Backbone Comparison"
Write-Host "  6A - Encoding Comparison (17 encodings)"
Write-Host "  5E - Two-Branch Fusion"
Write-Host "  5J - Training Enhancements"
Write-Host "  5K - TS Augmentation"
Write-Host "  5H - Resolution"
Write-Host "  5M - Transfer Learning"
Write-Host "  5L - Ensemble"
Write-Host ""
Write-Host "Remaining for A100:"
Write-Host "  5F  - Dataset Scale (216 runs)"
Write-Host "  5O  - Full 32-dataset benchmark"
Write-Host "  ViT experiments"
