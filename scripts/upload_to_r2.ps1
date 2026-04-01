# ============================================================
# Upload chart images and results to Cloudflare R2
# ============================================================
# Prerequisites:
#   pip install awscli
#   Set environment variables:
#     $env:R2_ACCOUNT_ID = "your_account_id"
#     $env:R2_ACCESS_KEY = "your_access_key"
#     $env:R2_SECRET_KEY = "your_secret_key"
#     $env:R2_BUCKET = "vtbench"
# ============================================================

$R2_ENDPOINT = "https://$env:R2_ACCOUNT_ID.r2.cloudflarestorage.com"

# Configure AWS CLI for R2
$env:AWS_ACCESS_KEY_ID = $env:R2_ACCESS_KEY
$env:AWS_SECRET_ACCESS_KEY = $env:R2_SECRET_KEY
$env:AWS_DEFAULT_REGION = "auto"

Write-Host "=== Uploading chart images to R2 ===" -ForegroundColor Cyan

# Upload chart images (D:\chart_images -> r2://vtbench/chart_images/)
Write-Host "Uploading chart images..."
aws s3 sync "D:\chart_images" "s3://$env:R2_BUCKET/chart_images/" `
    --endpoint-url $R2_ENDPOINT `
    --only-show-errors

# Upload results
Write-Host "Uploading results..."
aws s3 sync "results/" "s3://$env:R2_BUCKET/results/" `
    --endpoint-url $R2_ENDPOINT `
    --only-show-errors

# Upload code (for A100 deployment)
Write-Host "Uploading codebase..."
aws s3 sync "." "s3://$env:R2_BUCKET/code/" `
    --endpoint-url $R2_ENDPOINT `
    --exclude "*.pyc" --exclude "__pycache__/*" `
    --exclude "chart_images/*" --exclude "UCRArchive_2018/*" `
    --exclude ".git/*" --exclude "results/*" `
    --only-show-errors

Write-Host ""
Write-Host "=== Upload complete ===" -ForegroundColor Green
Write-Host "Images: s3://$env:R2_BUCKET/chart_images/"
Write-Host "Results: s3://$env:R2_BUCKET/results/"
Write-Host "Code: s3://$env:R2_BUCKET/code/"
