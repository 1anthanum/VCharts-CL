# ============================================================
# VTBench Top Monitor - GPU + CPU + Experiment Progress
# ============================================================
# Usage: powershell -ExecutionPolicy Bypass -File scripts/watch_top.ps1
# Exit:  Ctrl+C
# ============================================================

param(
    [int]$RefreshSeconds = 3
)

function Get-GpuInfo {
    try {
        $lines = nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
        if ($lines) {
            $parts = $lines.Split(',').Trim()
            return @{
                Name   = $parts[0]
                GpuUtil = "$($parts[1])%"
                MemUsed = "$($parts[2]) MB"
                MemTotal = "$($parts[3]) MB"
                MemPct  = [math]::Round([int]$parts[2] / [int]$parts[3] * 100, 1)
                Temp    = "$($parts[4])C"
                Power   = "$($parts[5])W"
            }
        }
    } catch { return $null }
}

function Get-PythonProcesses {
    Get-Process python -ErrorAction SilentlyContinue |
        Select-Object Id,
            @{N='CPU_s'; E={[math]::Round($_.CPU, 1)}},
            @{N='Mem_MB'; E={[math]::Round($_.WorkingSet64 / 1MB, 0)}},
            @{N='Runtime'; E={
                $span = (Get-Date) - $_.StartTime
                "{0:00}:{1:00}:{2:00}" -f [int]$span.TotalHours, $span.Minutes, $span.Seconds
            }},
            @{N='CmdLine'; E={
                try {
                    $wmi = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.Id)" -ErrorAction SilentlyContinue
                    $cmd = $wmi.CommandLine
                    if ($cmd.Length -gt 80) { $cmd.Substring(0, 77) + "..." } else { $cmd }
                } catch { "python" }
            }} |
        Sort-Object Mem_MB -Descending
}

function Get-ExperimentProgress {
    $logDir = "results"
    $logs = @(
        @{Name="6B"; File="exp6b.log"},
        @{Name="6A"; File="exp6a.log"},
        @{Name="B2"; File="exp_b2_chain.log"},
        @{Name="5F"; File="exp5f_v2.log"},
        @{Name="5J"; File="exp5j.log"},
        @{Name="5K"; File="exp5k.log"},
        @{Name="5H"; File="exp5h.log"},
        @{Name="5G"; File="exp5g_v2.log"},
        @{Name="5M"; File="exp5m.log"},
        @{Name="5L"; File="exp5l.log"}
    )

    $results = @()
    foreach ($log in $logs) {
        $path = Join-Path $logDir $log.File
        if (Test-Path $path) {
            $info = Get-Item $path
            $sizeKB = [math]::Round($info.Length / 1KB, 1)
            $age = (Get-Date) - $info.LastWriteTime
            $lastLine = ""
            try { $lastLine = (Get-Content $path -Tail 1 -ErrorAction SilentlyContinue) } catch {}
            # Detect completion
            $status = "RUNNING"
            if ($age.TotalMinutes -gt 10) { $status = "STALE?" }
            $content = Get-Content $path -Raw -ErrorAction SilentlyContinue
            if ($content -match "COMPLETE|All.*done|Saved.*summary") { $status = "DONE" }
            if ($content -match "Error|Traceback") { $status = "ERROR!" }

            $results += [PSCustomObject]@{
                Exp    = $log.Name
                Status = $status
                Size   = "${sizeKB}KB"
                Age    = "{0:00}:{1:00}" -f [int]$age.TotalMinutes, $age.Seconds
                Last   = if ($lastLine.Length -gt 60) { $lastLine.Substring(0,57) + "..." } else { $lastLine }
            }
        }
    }
    return $results
}

# Main loop
while ($true) {
    Clear-Host

    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "=== VTBench Monitor === $now === Refresh: ${RefreshSeconds}s ===" -ForegroundColor Cyan
    Write-Host ""

    # GPU
    $gpu = Get-GpuInfo
    if ($gpu) {
        $memBar = "#" * [math]::Floor($gpu.MemPct / 5) + "." * (20 - [math]::Floor($gpu.MemPct / 5))
        Write-Host "GPU: $($gpu.Name)" -ForegroundColor Green
        Write-Host "  Util: $($gpu.GpuUtil)  |  Mem: $($gpu.MemUsed)/$($gpu.MemTotal) ($($gpu.MemPct)%)  |  Temp: $($gpu.Temp)  |  Power: $($gpu.Power)"
        Write-Host "  VRAM: [$memBar] $($gpu.MemPct)%" -ForegroundColor $(if([int]$gpu.MemPct -gt 85){'Red'}elseif([int]$gpu.MemPct -gt 60){'Yellow'}else{'Green'})
    } else {
        Write-Host "GPU: N/A" -ForegroundColor Red
    }
    Write-Host ""

    # CPU + Memory
    $cpuLoad = (Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue).LoadPercentage
    $mem = Get-CimInstance Win32_OperatingSystem -ErrorAction SilentlyContinue
    if ($mem) {
        $totalGB = [math]::Round($mem.TotalVisibleMemorySize / 1MB, 1)
        $freeGB = [math]::Round($mem.FreePhysicalMemory / 1MB, 1)
        $usedGB = $totalGB - $freeGB
        $memPct = [math]::Round($usedGB / $totalGB * 100, 1)
        Write-Host "CPU: ${cpuLoad}%  |  RAM: ${usedGB}/${totalGB} GB (${memPct}%)" -ForegroundColor Green
    }
    Write-Host ""

    # Python processes
    $procs = Get-PythonProcesses
    if ($procs) {
        Write-Host "Python Processes ($($procs.Count)):" -ForegroundColor Yellow
        Write-Host ("{0,-8} {1,-8} {2,-10} {3,-10} {4}" -f "PID", "CPU(s)", "Mem(MB)", "Runtime", "Command")
        Write-Host ("-" * 90)
        foreach ($p in $procs) {
            Write-Host ("{0,-8} {1,-8} {2,-10} {3,-10} {4}" -f $p.Id, $p.CPU_s, $p.Mem_MB, $p.Runtime, $p.CmdLine)
        }
    } else {
        Write-Host "No Python processes running." -ForegroundColor DarkGray
    }
    Write-Host ""

    # Experiment progress
    $exps = Get-ExperimentProgress
    if ($exps) {
        Write-Host "Experiment Logs:" -ForegroundColor Yellow
        Write-Host ("{0,-6} {1,-8} {2,-10} {3,-8} {4}" -f "Exp", "Status", "Size", "Age", "Last Line")
        Write-Host ("-" * 90)
        foreach ($e in $exps) {
            $color = switch ($e.Status) {
                "DONE"    { "Green" }
                "RUNNING" { "White" }
                "STALE?"  { "Yellow" }
                "ERROR!"  { "Red" }
                default   { "Gray" }
            }
            Write-Host ("{0,-6} {1,-8} {2,-10} {3,-8} {4}" -f $e.Exp, $e.Status, $e.Size, $e.Age, $e.Last) -ForegroundColor $color
        }
    }

    Write-Host ""
    Write-Host "Press Ctrl+C to exit" -ForegroundColor DarkGray

    Start-Sleep -Seconds $RefreshSeconds
}
