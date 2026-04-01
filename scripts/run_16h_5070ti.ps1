# ============================================================
# VTBench 16-Hour Optimized Plan - RTX 5070 Ti (16GB VRAM)
# ============================================================
#
# Hardware: RTX 5070 Ti 16GB GDDR7, PyTorch 2.2.2 (PTX fallback)
# Strategy: 2 GPU experiments in parallel (~5-6GB each @ bs=32)
#           + CPU image gen in background
#
# Phases:
#   A  (0:00-0:45)   Image generation (CPU only)
#   B1 (0:45-1:15)   6B numerical baselines (GPU-light warmup)
#   B2 (1:15-7:00)   Parallel GPU pairs: 6A + (5I->5E->5B)
#   B3 (7:00-9:30)   Parallel GPU pairs: 5F + 5J
#   B4 (9:30-11:30)  Parallel GPU pairs: 5K + 5H
#   B5 (11:30-13:00) Parallel GPU pairs: 5G + 5M
#   B6 (13:00-14:00) 5L ensemble (needs previous models)
#   Buffer (14:00-16:00) Re-runs / overflow
#
# Total: 12 experiments, ~1,800 runs, estimated 11-13h
# ============================================================

$ErrorActionPreference = "Continue"
$env:CHART_IMAGE_ROOT = "D:\chart_images"
$LOGDIR = "results"
$SCRIPTDIR = "scripts"

function Log-Time { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]" }

function Run-Parallel {
    param(
        [string]$Name1,
        [string]$Cmd1,
        [string]$Log1,
        [string]$Name2,
        [string]$Cmd2,
        [string]$Log2
    )
    Write-Host ""
    Write-Host "$(Log-Time) >>> PARALLEL: $Name1 + $Name2" -ForegroundColor Magenta
    Write-Host "$(Log-Time)     GPU1: $Name1 -> $Log1" -ForegroundColor Yellow
    Write-Host "$(Log-Time)     GPU2: $Name2 -> $Log2" -ForegroundColor Yellow

    $job1 = Start-Job -ScriptBlock {
        param($cmd, $logfile, $workdir, $chartroot)
        Set-Location $workdir
        $env:CHART_IMAGE_ROOT = $chartroot
        $env:CUDA_MPS_PIPE_DIRECTORY = ""  # avoid MPS conflicts
        Invoke-Expression "$cmd 2>&1" | Out-File -FilePath $logfile -Encoding utf8
    } -ArgumentList $Cmd1, "$LOGDIR/$Log1", $PWD.Path, "D:\chart_images"

    $job2 = Start-Job -ScriptBlock {
        param($cmd, $logfile, $workdir, $chartroot)
        Set-Location $workdir
        $env:CHART_IMAGE_ROOT = $chartroot
        $env:CUDA_MPS_PIPE_DIRECTORY = ""
        Invoke-Expression "$cmd 2>&1" | Out-File -FilePath $logfile -Encoding utf8
    } -ArgumentList $Cmd2, "$LOGDIR/$Log2", $PWD.Path, "D:\chart_images"

    Write-Host "$(Log-Time)     Jobs started: $($job1.Id), $($job2.Id)"
    Wait-Job $job1, $job2 | Out-Null

    $state1 = $job1.State
    $state2 = $job2.State
    Write-Host "$(Log-Time)     $Name1 : $state1" -ForegroundColor $(if($state1 -eq 'Completed'){'Green'}else{'Red'})
    Write-Host "$(Log-Time)     $Name2 : $state2" -ForegroundColor $(if($state2 -eq 'Completed'){'Green'}else{'Red'})

    # Capture any errors
    if ($job1.State -eq 'Failed') { Receive-Job $job1 -ErrorAction SilentlyContinue | Out-File "$LOGDIR/${Log1}.errors" -Encoding utf8 }
    if ($job2.State -eq 'Failed') { Receive-Job $job2 -ErrorAction SilentlyContinue | Out-File "$LOGDIR/${Log2}.errors" -Encoding utf8 }

    Remove-Job $job1, $job2 -Force
}

Write-Host ""
Write-Host "$(Log-Time) ========================================================" -ForegroundColor Cyan
Write-Host "$(Log-Time)  VTBench 16-Hour Optimized Plan - RTX 5070 Ti (16GB)   " -ForegroundColor Cyan
Write-Host "$(Log-Time)  Strategy: 2x parallel GPU + background CPU image gen   " -ForegroundColor Cyan
Write-Host "$(Log-Time)  Experiments: 6B,6A,5I,5E,5B,5F,5G,5H,5J,5K,5M,5L     " -ForegroundColor Cyan
Write-Host "$(Log-Time) ========================================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# PHASE A: Image Generation (CPU only, ~45 min)
# ============================================================
Write-Host "$(Log-Time) === PHASE A: Image Generation (CPU) ===" -ForegroundColor Green

Write-Host "$(Log-Time) A1: Mathematical encodings for 7 core datasets..."
python $SCRIPTDIR/pregenerate_all_encodings.py `
    --datasets SyntheticControl GunPoint CBF Trace ECG5000 FordA Wafer `
    --encodings gasf gadf mtf mtf_16 rp rp_grayscale cwt stft `
    --rgb-presets gasf_gadf_mtf rp_cwt_stft `
    --colormaps viridis plasma inferno `
    --image-size 128
Write-Host "$(Log-Time) A1 done."

Write-Host "$(Log-Time) A2: Chart images for all experiment datasets..."
python $SCRIPTDIR/pregenerate_all_images.py `
    --datasets SyntheticControl GunPoint CBF Trace ECG5000 FordA Wafer `
        TwoPatterns BeetleFly TwoLeadECG Coffee Lightning2 ECG200 `
        Computers Yoga `
    --chart-types line area bar scatter `
    --label-modes with_label
Write-Host "$(Log-Time) A2 done."

Write-Host "$(Log-Time) === PHASE A COMPLETE ===" -ForegroundColor Green
Write-Host ""

# --- Background: Extended dataset image gen (CPU, non-blocking) ---
Write-Host "$(Log-Time) Starting background image gen for extended datasets..."
$bgImageGen = Start-Job -ScriptBlock {
    param($workdir, $chartroot)
    Set-Location $workdir
    $env:CHART_IMAGE_ROOT = $chartroot
    python scripts/pregenerate_all_encodings.py `
        --all-datasets `
        --encodings gasf gadf mtf rp cwt stft `
        --rgb-presets gasf_gadf_mtf `
        --image-size 128 2>&1
} -ArgumentList $PWD.Path, "D:\chart_images"
Write-Host "$(Log-Time) Background gen job: $($bgImageGen.Id)"

# ============================================================
# PHASE B1: Warmup - 6B Numerical Baselines (solo, ~30 min)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === B1: 6B Numerical Baselines (warmup, solo) ===" -ForegroundColor Yellow
python $SCRIPTDIR/experiment_6b_numerical_baseline.py `
    --config vtbench/config/experiment_6b_numerical.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp6b.log"
Write-Host "$(Log-Time) 6B COMPLETE." -ForegroundColor Green

# ============================================================
# PHASE B2: 6A (long, ~5-6h) + sequential 5I->5E->5B (short chain, ~4.5h)
# ============================================================
# 6A is the longest experiment. We pair it with a chain of shorter ones.
# Both share the GPU - 6A ~5GB + chain ~5GB = ~10GB < 16GB

Write-Host ""
Write-Host "$(Log-Time) === B2: PARALLEL - 6A (long) + 5I->5E->5B chain ===" -ForegroundColor Magenta

# GPU Slot 1: 6A Encoding Comparison (the key experiment, ~5-6h)
$gpu1_6a = Start-Job -ScriptBlock {
    param($workdir, $chartroot)
    Set-Location $workdir
    $env:CHART_IMAGE_ROOT = $chartroot
    python scripts/experiment_6a_encodings.py `
        --config vtbench/config/experiment_6a_encodings.yaml 2>&1
} -ArgumentList $PWD.Path, "D:\chart_images"

# GPU Slot 2: 5I -> 5E -> 5B chain (~1.5h + 2h + 1h = 4.5h)
$gpu2_chain = Start-Job -ScriptBlock {
    param($workdir, $chartroot)
    Set-Location $workdir
    $env:CHART_IMAGE_ROOT = $chartroot

    Write-Output "=== 5I Backbones ==="
    python scripts/experiment_5i_backbones.py `
        --config vtbench/config/experiment_5i_backbones.yaml 2>&1

    Write-Output "=== 5E Two-Branch ==="
    python scripts/experiment_5e_two_branch.py `
        --config vtbench/config/experiment_5e_full.yaml 2>&1

    Write-Output "=== 5B Multi-Chart ==="
    python scripts/experiment_5b_multi_chart.py `
        --config vtbench/config/experiment_5b_full.yaml 2>&1
} -ArgumentList $PWD.Path, "D:\chart_images"

Write-Host "$(Log-Time)   GPU1: 6A encoding comparison (Job $($gpu1_6a.Id))"
Write-Host "$(Log-Time)   GPU2: 5I->5E->5B chain (Job $($gpu2_chain.Id))"

# Wait for both
Wait-Job $gpu1_6a, $gpu2_chain | Out-Null

# Capture output to logs
Receive-Job $gpu1_6a 2>&1 | Out-File "$LOGDIR/exp6a.log" -Encoding utf8
Receive-Job $gpu2_chain 2>&1 | Out-File "$LOGDIR/exp_b2_chain.log" -Encoding utf8

Write-Host "$(Log-Time)   6A: $($gpu1_6a.State)" -ForegroundColor $(if($gpu1_6a.State -eq 'Completed'){'Green'}else{'Red'})
Write-Host "$(Log-Time)   5I+5E+5B: $($gpu2_chain.State)" -ForegroundColor $(if($gpu2_chain.State -eq 'Completed'){'Green'}else{'Red'})
Remove-Job $gpu1_6a, $gpu2_chain -Force

Write-Host "$(Log-Time) B2 COMPLETE." -ForegroundColor Green

# ============================================================
# PHASE B3: 5F (heavy, ~2h) || 5J (medium, ~1.5h)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === B3: PARALLEL - 5F Scale + 5J Training ===" -ForegroundColor Magenta
Run-Parallel `
    -Name1 "5F Dataset Scale" `
    -Cmd1 "python scripts/experiment_5f_scale.py --config vtbench/config/experiment_5f_scale.yaml" `
    -Log1 "exp5f_v2.log" `
    -Name2 "5J Training Strategies" `
    -Cmd2 "python scripts/experiment_5j_training.py --config vtbench/config/experiment_5j_training.yaml" `
    -Log2 "exp5j.log"
Write-Host "$(Log-Time) B3 COMPLETE." -ForegroundColor Green

# ============================================================
# PHASE B4: 5K (medium, ~1.5h) || 5H (medium, ~1h)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === B4: PARALLEL - 5K TS Augment + 5H Resolution ===" -ForegroundColor Magenta
Run-Parallel `
    -Name1 "5K TS Augmentation" `
    -Cmd1 "python scripts/experiment_5k_ts_augment.py --config vtbench/config/experiment_5k_ts_augment.yaml" `
    -Log1 "exp5k.log" `
    -Name2 "5H Resolution" `
    -Cmd2 "python scripts/experiment_5h_resolution.py --config vtbench/config/experiment_5h_resolution.yaml" `
    -Log2 "exp5h.log"
Write-Host "$(Log-Time) B4 COMPLETE." -ForegroundColor Green

# ============================================================
# PHASE B5: 5G (light, ~30min) || 5M (medium, ~1h)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === B5: PARALLEL - 5G Chart Type + 5M Transfer ===" -ForegroundColor Magenta
Run-Parallel `
    -Name1 "5G Chart Type Comparison" `
    -Cmd1 "python scripts/experiment_5g_chart_type.py --config vtbench/config/experiment_5g_chart_type.yaml" `
    -Log1 "exp5g_v2.log" `
    -Name2 "5M Transfer Learning" `
    -Cmd2 "python scripts/experiment_5m_transfer.py --config vtbench/config/experiment_5m_transfer.yaml" `
    -Log2 "exp5m.log"
Write-Host "$(Log-Time) B5 COMPLETE." -ForegroundColor Green

# ============================================================
# PHASE B6: 5L Ensemble (needs previous models, solo, ~30 min)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) === B6: 5L Ensemble (solo, depends on prior models) ===" -ForegroundColor Yellow
python $SCRIPTDIR/experiment_5l_ensemble.py `
    --config vtbench/config/experiment_5l_ensemble.yaml `
    2>&1 | Tee-Object -FilePath "$LOGDIR/exp5l.log"
Write-Host "$(Log-Time) B6 COMPLETE." -ForegroundColor Green

# ============================================================
# Cleanup: Wait for background image gen
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) Checking background image generation..." -ForegroundColor Cyan
if ($bgImageGen -and $bgImageGen.State -eq "Running") {
    Write-Host "$(Log-Time) Background gen still running. Waiting..."
    Wait-Job $bgImageGen | Out-Null
}
if ($bgImageGen) {
    Receive-Job $bgImageGen -ErrorAction SilentlyContinue | Out-File "$LOGDIR/bg_imagegen.log" -Encoding utf8
    Remove-Job $bgImageGen -Force -ErrorAction SilentlyContinue
}
Write-Host "$(Log-Time) Background image generation done."

# ============================================================
# SUMMARY
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) ========================================================" -ForegroundColor Cyan
Write-Host "$(Log-Time)  ALL 12 EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host "$(Log-Time) ========================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Completed:"
Write-Host "  Phase B1: 6B  Numerical Baselines        (63 runs)"
Write-Host "  Phase B2: 6A  Encoding Comparison         (714 runs)  ** KEY **"
Write-Host "            5I  Backbone Comparison          (126 runs)"
Write-Host "            5E  Two-Branch Fusion            (300 runs)"
Write-Host "            5B  Multi-Chart Fusion           (90 runs)"
Write-Host "  Phase B3: 5F  Dataset Scale                (216 runs)"
Write-Host "            5J  Training Strategies          (180 runs)"
Write-Host "  Phase B4: 5K  TS Augmentation              (120 runs)"
Write-Host "            5H  Image Resolution             (63 runs)"
Write-Host "  Phase B5: 5G  Chart Type Comparison        (96 runs)"
Write-Host "            5M  Transfer Learning            (~40 runs)"
Write-Host "  Phase B6: 5L  Ensemble                     (~30 runs)"
Write-Host ""
Write-Host "  Total: ~2,038 training runs"
Write-Host ""
Write-Host "Previously completed (no re-run needed):"
Write-Host "  5A  Augmentation Robustness"
Write-Host "  5C  ResNet18 Pretrained"
Write-Host "  5D  Rendering Variants"
Write-Host ""
Write-Host "Log files in: $LOGDIR/exp*.log"
Write-Host "$(Log-Time) Done."
