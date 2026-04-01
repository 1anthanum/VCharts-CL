# ============================================================
# VTBench — One-Command Launcher (RTX 5070 Ti 16GB)
# ============================================================
#
# Usage:
#   .\run.ps1                  # full suite
#   .\run.ps1 -Resume          # skip completed experiments
#   .\run.ps1 -Phase B2        # start from phase B2
#   .\run.ps1 -Only 6b,6a      # run specific experiments
#   .\run.ps1 -DryRun          # show plan only
#   .\run.ps1 -Solo            # no parallelism
#
# ============================================================

param(
    [switch]$Resume,
    [switch]$DryRun,
    [switch]$Solo,
    [string]$Phase = "",
    [string]$Only = ""
)

$ErrorActionPreference = "Stop"

# ---- Environment ----
$env:CHART_IMAGE_ROOT = "D:\chart_images"
$env:PYTHONUNBUFFERED = "1"

# ---- Verify ----
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  VTBench Experiment Suite" -ForegroundColor Cyan
Write-Host "  RTX 5070 Ti 16GB | batch_size=32" -ForegroundColor Cyan
Write-Host "  PyTorch 2.12+cu128 | W&B 0.25.1" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[Check] PyTorch + CUDA..." -ForegroundColor Yellow
python -c "import torch; v=torch.__version__; c=torch.cuda.is_available(); g=torch.cuda.get_device_name(0) if c else 'N/A'; print(f'  PyTorch {v}  CUDA: {c}  GPU: {g}')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FATAL: PyTorch check failed" -ForegroundColor Red
    exit 1
}

Write-Host "[Check] W&B..." -ForegroundColor Yellow
python -c "import wandb; print(f'  wandb {wandb.__version__}')"

Write-Host "[Check] CHART_IMAGE_ROOT = $env:CHART_IMAGE_ROOT" -ForegroundColor Yellow
if (-not (Test-Path $env:CHART_IMAGE_ROOT)) {
    Write-Host "  Creating $env:CHART_IMAGE_ROOT..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path $env:CHART_IMAGE_ROOT -Force | Out-Null
}
Write-Host ""

# ---- Build args ----
$args_list = @()
if ($Resume)  { $args_list += "--resume" }
if ($DryRun)  { $args_list += "--dry-run" }
if ($Solo)    { $args_list += "--solo" }
if ($Phase)   { $args_list += "--phase"; $args_list += $Phase }
if ($Only)    { $args_list += "--only"; $args_list += ($Only -split ",") }

# ---- Run ----
$startTime = Get-Date
Write-Host "Starting: python scripts/orchestrator.py $($args_list -join ' ')" -ForegroundColor Green
Write-Host "Time: $($startTime.ToString('yyyy-MM-dd HH:mm:ss'))" -ForegroundColor Green
Write-Host ""

python scripts/orchestrator.py @args_list

$elapsed = (Get-Date) - $startTime
Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Finished in $($elapsed.ToString('hh\:mm\:ss'))" -ForegroundColor Cyan
Write-Host "  Logs: results/logs/" -ForegroundColor Cyan
Write-Host "  State: results/orchestrator_state.json" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
