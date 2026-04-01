"""
GPU Diagnosis Script for RTX 5070 Ti Crash Investigation
=========================================================
Run on Windows: python scripts/gpu_diagnosis.py

Checks:
  1. nvidia-smi full device query (XID errors, ECC, PCIe, clocks, thermals)
  2. Windows Event Viewer for display driver crashes (TDR / nvlddmkm)
  3. CUDA stress test (allocate + compute to provoke failures)
  4. PCIe link width/gen verification
  5. Driver version cross-reference

No dependencies beyond PyTorch (already installed).
"""

import subprocess
import sys
import os
import json
import time
import datetime
import re
import platform

DIVIDER = "=" * 70
SECTION = "-" * 50


def header(title):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def run_cmd(cmd, shell=False, timeout=30):
    """Run a command and return stdout, stderr, returncode."""
    try:
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, shell=shell
        )
        return r.stdout.strip(), r.stderr.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except FileNotFoundError:
        return "", "NOT_FOUND", -1


def check_nvidia_smi_basic():
    header("1. NVIDIA-SMI Basic Info")
    out, err, rc = run_cmd(["nvidia-smi"])
    if rc != 0:
        print(f"  [FAIL] nvidia-smi not available or GPU not responding: {err}")
        return False
    print(out)
    return True


def check_xid_errors():
    header("2. XID Errors & ECC Status")
    print("  XID errors indicate specific GPU failure modes:")
    print("    XID 13  = Graphics engine exception")
    print("    XID 31  = GPU memory page fault (BAD)")
    print("    XID 43  = GPU stopped processing")
    print("    XID 45  = Preemption failure")
    print("    XID 48  = Double-bit ECC error (VRAM defect)")
    print("    XID 61  = Internal microcontroller halt")
    print("    XID 62  = Internal microcontroller breakpoint")
    print("    XID 79  = GPU fell off the bus (PCIe FATAL)")
    print("    XID 119 = GSP firmware error")
    print(SECTION)

    # ECC status
    out, err, rc = run_cmd(["nvidia-smi", "-q", "-d", "ECC"])
    if rc == 0:
        print("[ECC Status]")
        print(out)
    else:
        print(f"  ECC query returned: {err}")

    # Page retirement (defective VRAM pages)
    out, err, rc = run_cmd(["nvidia-smi", "-q", "-d", "PAGE_RETIREMENT"])
    if rc == 0:
        print("\n[Page Retirement (Defective VRAM)]")
        print(out)
    else:
        print(f"  Page retirement query returned: {err}")

    # Inforom check
    out, err, rc = run_cmd(["nvidia-smi", "-q", "-d", "INFOROM"])
    if rc == 0 and "Corrupt" in out:
        print("\n  [WARNING] InfoROM corruption detected — card may need RMA")


def check_pcie():
    header("3. PCIe Link Status")
    out, err, rc = run_cmd(["nvidia-smi", "-q", "-d", "PCIE"])
    if rc == 0:
        print(out)
        # Parse key values
        gen_match = re.findall(r"(Max|Current)\s+Link\s+Gen\s*:\s*(\d+)", out, re.IGNORECASE)
        width_match = re.findall(r"(Max|Current)\s+Link\s+Width\s*:\s*(\w+)", out, re.IGNORECASE)
        if gen_match:
            print(f"\n  [SUMMARY]")
            for label, val in gen_match:
                print(f"    {label} Link Gen: {val}")
            for label, val in width_match:
                print(f"    {label} Link Width: {val}")
            # Check for degraded link
            gens = {label.lower(): int(val) for label, val in gen_match}
            widths = {label.lower(): val for label, val in width_match}
            if gens.get("current", 0) < gens.get("max", 0):
                print("  [WARNING] PCIe link running below max generation!")
                print("  → This can cause instability. Check BIOS settings.")
            if widths.get("current", "x16") != widths.get("max", "x16"):
                print("  [WARNING] PCIe link width degraded!")
                print("  → Card may not be fully seated or slot is damaged.")
    else:
        print(f"  PCIe query failed: {err}")


def check_clocks_and_power():
    header("4. Clock Speeds, Power & Thermal")
    fields = [
        "gpu_name", "driver_version", "vbios_version",
        "temperature.gpu", "temperature.memory",
        "power.draw", "power.limit", "power.default_limit", "power.max_limit",
        "clocks.current.graphics", "clocks.max.graphics",
        "clocks.current.memory", "clocks.max.memory",
        "clocks.current.sm",
        "fan.speed",
        "utilization.gpu", "utilization.memory",
        "memory.used", "memory.total",
        "pstate",
    ]
    query = ",".join(fields)
    out, err, rc = run_cmd([
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader"
    ])
    if rc == 0 and out:
        values = [v.strip() for v in out.split(",")]
        print(f"  GPU Name:             {values[0]}")
        print(f"  Driver Version:       {values[1]}")
        print(f"  VBIOS Version:        {values[2]}")
        print(f"  GPU Temp:             {values[3]} °C")
        print(f"  Memory Temp:          {values[4]} °C")
        print(f"  Power Draw:           {values[5]} W")
        print(f"  Power Limit:          {values[6]} W")
        print(f"  Default Power Limit:  {values[7]} W")
        print(f"  Max Power Limit:      {values[8]} W")
        print(f"  GPU Clock (cur/max):  {values[9]} / {values[10]} MHz")
        print(f"  Mem Clock (cur/max):  {values[11]} / {values[12]} MHz")
        print(f"  SM Clock:             {values[13]} MHz")
        print(f"  Fan Speed:            {values[14]}")
        print(f"  GPU Util:             {values[15]}")
        print(f"  Mem Util:             {values[16]}")
        print(f"  VRAM Used/Total:      {values[17]} / {values[18]}")
        print(f"  Performance State:    {values[19]}")

        # Check power limit vs default
        try:
            p_limit = float(values[6].replace(" W", "").strip())
            p_default = float(values[7].replace(" W", "").strip())
            if p_limit < p_default * 0.9:
                print(f"\n  [NOTE] Power limit is {p_limit}W vs default {p_default}W")
                print("  → You've lowered the power limit. This is fine for stability testing.")
        except (ValueError, IndexError):
            pass
    else:
        print(f"  Query failed: {err}")


def check_windows_events():
    header("5. Windows Event Viewer — Display Driver Crashes")
    if platform.system() != "Windows":
        print("  [SKIP] Not running on Windows")
        return

    # Check for nvlddmkm and display driver errors in System log
    ps_cmd = '''
    try {
        $events = Get-WinEvent -FilterHashtable @{LogName='System'; Level=1,2,3; StartTime=(Get-Date).AddDays(-7)} -MaxEvents 500 -ErrorAction Stop |
            Where-Object { $_.Message -match 'nvlddmkm|display driver|nvidia|GPU|TDR|video scheduler|dxgkrnl|dxgmms' } |
            Select-Object -First 30 TimeCreated, Id, LevelDisplayName, ProviderName,
                @{N='ShortMessage';E={$_.Message.Substring(0, [Math]::Min(300, $_.Message.Length))}}
        if ($events) {
            $events | Format-List
        } else {
            Write-Output "NO_GPU_EVENTS_FOUND"
        }
    } catch {
        Write-Output "EVENT_LOG_ERROR: $_"
    }
    '''
    out, err, rc = run_cmd(["powershell", "-Command", ps_cmd], timeout=30)

    if "NO_GPU_EVENTS_FOUND" in out:
        print("  [OK] No GPU-related error events in the last 7 days.")
        print("  → If crashes happened, they may not have been logged (hard power-off)")
    elif "EVENT_LOG_ERROR" in out:
        print(f"  [WARN] Could not read Event Log (may need admin): {out}")
        print("  → Try: right-click → Run as Administrator")
    elif out:
        print("  [FOUND] GPU-related events in the last 7 days:\n")
        print(out)
        # Count severity
        critical = out.lower().count("critical")
        errors = out.lower().count("error")
        warnings = out.lower().count("warning")
        print(f"\n  Summary: {critical} critical, {errors} errors, {warnings} warnings")
        if "nvlddmkm" in out.lower():
            print("  → nvlddmkm errors confirm NVIDIA kernel driver crashes")
            print("  → Recommendation: DDU clean reinstall or try older driver")
        if "tdr" in out.lower() or "timeout detection" in out.lower():
            print("  → TDR events = GPU stopped responding and Windows reset it")
    else:
        print("  No output from Event Viewer query.")

    # Also check for LiveKernelEvent (BSOD-related)
    print(f"\n{SECTION}")
    print("  Checking for LiveKernelEvent (GPU-related BSOD/hang)...")
    ps_cmd2 = '''
    try {
        $lke = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-LiveDump','Microsoft-Windows-WER-SystemErrorReporting'; StartTime=(Get-Date).AddDays(-7)} -MaxEvents 10 -ErrorAction Stop
        if ($lke) {
            $lke | Select-Object TimeCreated, Id, ProviderName, @{N='Msg';E={$_.Message.Substring(0,[Math]::Min(200,$_.Message.Length))}} | Format-List
        } else {
            Write-Output "NO_LIVE_KERNEL_EVENTS"
        }
    } catch {
        Write-Output "NO_LIVE_KERNEL_EVENTS"
    }
    '''
    out2, err2, rc2 = run_cmd(["powershell", "-Command", ps_cmd2], timeout=20)
    if "NO_LIVE_KERNEL_EVENTS" in out2:
        print("  [OK] No LiveKernelEvent / BSOD reports in last 7 days.")
    elif out2:
        print("  [FOUND] Kernel crash/hang events:\n")
        print(out2)


def check_cuda_stress():
    header("6. CUDA Quick Stress Test")
    try:
        import torch
    except ImportError:
        print("  [SKIP] PyTorch not installed")
        return

    if not torch.cuda.is_available():
        print("  [FAIL] CUDA not available")
        return

    device = torch.device("cuda")
    print(f"  PyTorch:  {torch.__version__}")
    print(f"  CUDA:     {torch.version.cuda}")
    print(f"  Device:   {torch.cuda.get_device_name(0)}")
    print(f"  Arch:     {torch.cuda.get_device_capability(0)}")

    # Test 1: Basic allocation
    print(f"\n{SECTION}")
    print("  Test A: VRAM allocation ladder (256MB → 4GB)...")
    sizes_mb = [256, 512, 1024, 2048, 4096]
    for sz in sizes_mb:
        try:
            numel = sz * 1024 * 1024 // 4  # float32
            t = torch.randn(numel, device=device)
            # Force sync
            torch.cuda.synchronize()
            _ = t.sum().item()
            del t
            torch.cuda.empty_cache()
            print(f"    {sz:>5} MB: OK")
        except RuntimeError as e:
            print(f"    {sz:>5} MB: FAILED — {e}")
            break

    # Test 2: Compute stress (matmul)
    print(f"\n{SECTION}")
    print("  Test B: Matrix multiply stress (fp32 + fp16)...")
    for dtype, label in [(torch.float32, "FP32"), (torch.float16, "FP16")]:
        try:
            n = 4096
            a = torch.randn(n, n, device=device, dtype=dtype)
            b = torch.randn(n, n, device=device, dtype=dtype)
            torch.cuda.synchronize()

            start = time.time()
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Verify result is not NaN/Inf
            has_nan = torch.isnan(c).any().item()
            has_inf = torch.isinf(c).any().item()
            checksum = c.sum().item()

            status = "OK"
            if has_nan:
                status = "NaN DETECTED (possible VRAM corruption)"
            elif has_inf:
                status = "Inf DETECTED (overflow or corruption)"

            print(f"    {label} 4096×4096 ×10: {elapsed:.2f}s — {status}")
            del a, b, c
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"    {label}: FAILED — {e}")

    # Test 3: AMP training simulation (this is what crashes in your experiments)
    print(f"\n{SECTION}")
    print("  Test C: AMP training simulation (mimics your experiment pipeline)...")
    try:
        import torch.nn as nn
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.CrossEntropyLoss()

        print("    Running 100 AMP training steps...")
        for step in range(100):
            x = torch.randn(32, 3, 64, 64, device=device)
            y = torch.randint(0, 10, (32,), device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step + 1) % 25 == 0:
                torch.cuda.synchronize()
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"    Step {step+1}/100: loss={loss.item():.4f}, VRAM={mem:.0f}MB — OK")

        del model, optimizer, scaler
        torch.cuda.empty_cache()
        print("    [PASS] AMP training simulation completed without errors")

    except RuntimeError as e:
        print(f"    [FAIL] AMP training crashed: {e}")
        print("    → This reproduces your experiment crash pattern!")

    # Test 4: Sustained compute (longer stress)
    print(f"\n{SECTION}")
    print("  Test D: Sustained GPU compute (60 seconds)...")
    try:
        n = 2048
        a = torch.randn(n, n, device=device)
        b = torch.randn(n, n, device=device)

        start = time.time()
        iters = 0
        last_report = start
        while time.time() - start < 60:
            c = torch.matmul(a, b)
            iters += 1
            now = time.time()
            if now - last_report >= 15:
                torch.cuda.synchronize()
                temp_out, _, _ = run_cmd([
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,clocks.current.graphics",
                    "--format=csv,noheader,nounits"
                ])
                print(f"    {now - start:.0f}s: {iters} matmuls, GPU: {temp_out}")
                last_report = now

        torch.cuda.synchronize()
        elapsed = time.time() - start
        print(f"    [PASS] {iters} matmuls in {elapsed:.0f}s — GPU survived sustained load")
        del a, b, c
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"    [FAIL] Crashed after {time.time() - start:.0f}s: {e}")


def check_tdr_registry():
    header("7. TDR Registry Settings")
    if platform.system() != "Windows":
        print("  [SKIP] Not Windows")
        return

    ps_cmd = '''
    $path = "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers"
    $keys = @("TdrLevel", "TdrDelay", "TdrDdiDelay", "TdrLimitCount", "TdrLimitTime")
    foreach ($k in $keys) {
        $val = (Get-ItemProperty -Path $path -Name $k -ErrorAction SilentlyContinue).$k
        if ($val -ne $null) {
            Write-Output "${k}: $val"
        } else {
            Write-Output "${k}: (default/not set)"
        }
    }
    '''
    out, err, rc = run_cmd(["powershell", "-Command", ps_cmd], timeout=10)
    if out:
        print(out)
        print(f"\n  Reference defaults:")
        print(f"    TdrLevel:      3 (recovery enabled)")
        print(f"    TdrDelay:      2 seconds (time before TDR triggers)")
        print(f"    TdrDdiDelay:   5 seconds")
        print(f"    TdrLimitCount: 5 (resets before BSOD)")
        print(f"    TdrLimitTime:  60 seconds")
        if "TdrDelay" in out and "not set" not in out:
            print("\n  [NOTE] Custom TDR settings detected.")
    else:
        print("  Could not read TDR registry. Try running as Administrator.")


def check_hw_accel_status():
    header("8. Hardware Acceleration Settings")
    if platform.system() != "Windows":
        print("  [SKIP] Not Windows")
        return

    ps_cmd = '''
    # HW-accelerated GPU scheduling
    $hwsch = (Get-ItemProperty -Path "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers" -Name "HwSchMode" -ErrorAction SilentlyContinue).HwSchMode
    if ($hwsch -eq $null) { $hwsch = "(not set, default=ON on supported GPUs)" }
    elseif ($hwsch -eq 1) { $hwsch = "1 (OFF)" }
    elseif ($hwsch -eq 2) { $hwsch = "2 (ON)" }
    Write-Output "HW GPU Scheduling: $hwsch"

    # MPO (Multi-Plane Overlay) — known to cause issues
    $mpo = (Get-ItemProperty -Path "HKLM:\\SOFTWARE\\Microsoft\\Windows\\Dwm" -Name "OverlayTestMode" -ErrorAction SilentlyContinue).OverlayTestMode
    if ($mpo -eq $null) { $mpo = "(not set, MPO enabled)" }
    elseif ($mpo -eq 5) { $mpo = "5 (MPO DISABLED)" }
    Write-Output "Multi-Plane Overlay: $mpo"

    # Check DWM transparency
    $trans = (Get-ItemProperty -Path "HKCU:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize" -Name "EnableTransparency" -ErrorAction SilentlyContinue).EnableTransparency
    if ($trans -eq 0) { $trans = "0 (OFF)" }
    elseif ($trans -eq 1) { $trans = "1 (ON)" }
    else { $trans = "(unknown)" }
    Write-Output "Transparency Effects: $trans"
    '''
    out, err, rc = run_cmd(["powershell", "-Command", ps_cmd], timeout=10)
    if out:
        print(out)
    else:
        print("  Could not read settings. Try running as Administrator.")


def generate_recommendations():
    header("9. RECOMMENDATIONS")
    print("""
  Based on your crash pattern (low load, low temp, random timing) with
  RTX 5070 Ti + Driver 595.79 (2026-03-10):

  PRIORITY 1 — Driver (most likely cause):
    □ DDU clean reinstall in Safe Mode (wagnardsoft.com)
    □ Try the PREVIOUS Studio driver version (before 595.79)
    □ Or try Game Ready driver instead of Studio

  PRIORITY 2 — Windows settings:
    □ Disable HW GPU Scheduling:
      Settings → Display → Graphics → Change default graphics settings → OFF
    □ Disable Multi-Plane Overlay (known crash trigger):
      reg add "HKLM\\SOFTWARE\\Microsoft\\Windows\\Dwm" /v OverlayTestMode /t REG_DWORD /d 5 /f
    □ Close all browsers or disable their GPU acceleration

  PRIORITY 3 — TDR timeout (if crashes look like TDR):
    □ Increase TDR delay (gives GPU more time before Windows kills it):
      reg add "HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers" /v TdrDelay /t REG_DWORD /d 10 /f
      (Requires reboot)

  PRIORITY 4 — Hardware:
    □ Re-seat the GPU and 12V-2x6 power connector
    □ Check PCIe gen in BIOS — try forcing Gen4 instead of Gen5
    □ Run MemtestCL or OCCT VRAM test for 30+ minutes

  PRIORITY 5 — If nothing helps:
    □ Contact retailer for RMA — may be defective silicon
    □ File NVIDIA bug report: https://forms.gle/nvidia-bug-report

  FOR NOW — To run experiments more safely:
    □ Use: python scripts/orchestrator.py --resume
    □ The 3-layer monitoring is already in place
    □ All completed results are preserved
    """)


def main():
    print(DIVIDER)
    print("  RTX 5070 Ti GPU DIAGNOSIS")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  OS: {platform.platform()}")
    print(DIVIDER)

    if not check_nvidia_smi_basic():
        print("\n  [FATAL] Cannot communicate with GPU. Is the driver crashed right now?")
        print("  → Try rebooting, then run this script again.")
        return

    check_xid_errors()
    check_pcie()
    check_clocks_and_power()
    check_windows_events()
    check_tdr_registry()
    check_hw_accel_status()
    check_cuda_stress()
    generate_recommendations()

    print(f"\n{DIVIDER}")
    print("  DIAGNOSIS COMPLETE")
    print(f"  Run with admin rights for full Event Viewer access:")
    print(f"    Right-click terminal → Run as Administrator")
    print(f"    python scripts/gpu_diagnosis.py")
    print(DIVIDER)


if __name__ == "__main__":
    main()
