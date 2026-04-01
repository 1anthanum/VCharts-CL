# ============================================================
# Overnight Image Generation — Run before shutdown
# ============================================================
# Generates ALL encoding images on local CPU.
# No GPU needed. Can run while sleeping.
#
# Estimated time: 2-4 hours for core datasets
#                 6-10 hours for all 32 datasets
#
# After this completes, upload to R2, then use cloud GPU
# for training only (much cheaper).
# ============================================================

$ErrorActionPreference = "Continue"
$env:CHART_IMAGE_ROOT = "D:\chart_images"

function Log-Time { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]" }

Write-Host "$(Log-Time) === Overnight Image Generation Starting ===" -ForegroundColor Cyan
Write-Host ""

# ============================================================
# Step 1: Mathematical encodings for core 5 datasets (~30 min)
# ============================================================
Write-Host "$(Log-Time) Step 1: Core datasets — all mathematical encodings" -ForegroundColor Green
python scripts/pregenerate_all_encodings.py `
    --encodings gasf gadf mtf mtf_16 rp rp_grayscale cwt stft `
    --rgb-presets gasf_gadf_mtf rp_cwt_stft `
    --colormaps viridis plasma `
    --image-size 128
Write-Host "$(Log-Time) Step 1 complete." -ForegroundColor Green

# ============================================================
# Step 2: Extended datasets — same encodings (~1-2 hours)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) Step 2: Extended 15 datasets" -ForegroundColor Green
python scripts/pregenerate_all_encodings.py `
    --extended `
    --encodings gasf gadf mtf rp rp_grayscale cwt stft `
    --rgb-presets gasf_gadf_mtf `
    --no-colormaps `
    --image-size 128
Write-Host "$(Log-Time) Step 2 complete." -ForegroundColor Green

# ============================================================
# Step 3: ALL 32 datasets — core encodings only (~3-5 hours)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) Step 3: All 32 datasets — core encodings" -ForegroundColor Green
python scripts/pregenerate_all_encodings.py `
    --all-datasets `
    --encodings gasf gadf mtf rp cwt `
    --no-rgb --no-colormaps `
    --image-size 128
Write-Host "$(Log-Time) Step 3 complete." -ForegroundColor Green

# ============================================================
# Step 4: Chart images for extended datasets (if not cached)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) Step 4: Chart images for extended datasets" -ForegroundColor Green
python scripts/pregenerate_all_images.py `
    --chart-types line area bar scatter `
    --label-modes with_label
Write-Host "$(Log-Time) Step 4 complete." -ForegroundColor Green

# ============================================================
# Step 5: 10x TS augmentation images for 5K (~2 hours)
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) Step 5: 10x TS augmentation images" -ForegroundColor Green
python scripts/pregenerate_5k_images.py --multipliers 10
Write-Host "$(Log-Time) Step 5 complete." -ForegroundColor Green

# ============================================================
# Summary
# ============================================================
Write-Host ""
Write-Host "$(Log-Time) ============================================" -ForegroundColor Cyan
Write-Host "$(Log-Time) ALL IMAGE GENERATION COMPLETE!" -ForegroundColor Cyan
Write-Host "$(Log-Time) ============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Upload to R2: powershell -File scripts/upload_to_r2.ps1"
Write-Host "  2. Run training on cloud: bash scripts/run_a100.sh"
Write-Host ""

# Show disk usage
Write-Host "Disk usage:" -ForegroundColor Yellow
$size = (Get-ChildItem -Path "D:\chart_images" -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "  D:\chart_images: $([math]::Round($size, 2)) GB"

$imageCount = (Get-ChildItem -Path "D:\chart_images" -Recurse -Filter "*.png" | Measure-Object).Count
Write-Host "  Total images: $imageCount"
