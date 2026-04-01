# GPU Monitor - Refreshes every 2 seconds
# Usage: powershell -ExecutionPolicy Bypass -File scripts/watch_gpu.ps1
# Press Ctrl+C to stop

while ($true) {
    Clear-Host
    Write-Host "=== GPU Monitor === $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
    Write-Host ""
    nvidia-smi
    Write-Host ""
    Write-Host "=== Python Processes ===" -ForegroundColor Yellow
    Get-Process python* -ErrorAction SilentlyContinue |
        Select-Object Id, CPU,
            @{N='Mem(MB)';E={[math]::Round($_.WorkingSet64/1MB,0)}},
            @{N='Runtime';E={((Get-Date) - $_.StartTime).ToString('hh\:mm\:ss')}} |
        Format-Table -AutoSize
    Start-Sleep -Seconds 2
}
