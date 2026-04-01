#!/usr/bin/env python
"""
Layer 2: Independent GPU Watchdog Monitor
==========================================
Runs as a SEPARATE process from training. Logs GPU state every 5-10 seconds.
If the machine hard-crashes, this process dies too — but its last log entry
tells you the exact GPU state moments before the crash.

Usage:
    python scripts/watchdog.py                  # foreground, 10s interval
    python scripts/watchdog.py --interval 5     # 5-second interval
    python scripts/watchdog.py --background     # daemonize (writes to log only)

Output:
    results/watchdog/gpu_monitor.jsonl   (append-only, one JSON per line)
    results/watchdog/latest.json         (overwritten each tick for quick check)

Does NOT import torch or touch the GPU via CUDA — uses nvidia-smi only,
so it cannot interfere with training or cause CUDA context issues.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


WATCHDOG_DIR = os.path.join("results", "watchdog")
HEARTBEAT_DIR = os.path.join("results", "heartbeats")
LOG_FILE = os.path.join(WATCHDOG_DIR, "gpu_monitor.jsonl")
LATEST_FILE = os.path.join(WATCHDOG_DIR, "latest.json")


def get_gpu_stats():
    """Query nvidia-smi for GPU stats (no CUDA context needed)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw,power.limit,fan.speed,"
             "clocks.current.graphics,clocks.current.memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return {"error": f"nvidia-smi exit {result.returncode}"}

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 9:
            return {
                "gpu_util_pct": _safe_float(parts[0]),
                "mem_used_mb": _safe_float(parts[1]),
                "mem_total_mb": _safe_float(parts[2]),
                "temp_c": _safe_float(parts[3]),
                "power_w": _safe_float(parts[4]),
                "power_limit_w": _safe_float(parts[5]),
                "fan_pct": _safe_float(parts[6]),
                "clock_gpu_mhz": _safe_float(parts[7]),
                "clock_mem_mhz": _safe_float(parts[8]),
            }
        return {"error": f"unexpected output: {result.stdout.strip()}"}
    except FileNotFoundError:
        return {"error": "nvidia-smi not found"}
    except subprocess.TimeoutExpired:
        return {"error": "nvidia-smi timeout"}
    except Exception as e:
        return {"error": str(e)}


def get_training_pids():
    """Find running Python training processes."""
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                ["wmic", "process", "where",
                 "name='python.exe' or name='python3.exe'",
                 "get", "ProcessId,CommandLine", "/format:csv"],
                capture_output=True, text=True, timeout=10
            )
            lines = [l for l in result.stdout.strip().split("\n")
                     if "experiment_" in l or "simple_trainer" in l]
            return [l.strip() for l in lines[:5]]
        else:
            result = subprocess.run(
                ["ps", "aux"], capture_output=True, text=True, timeout=10
            )
            lines = [l for l in result.stdout.strip().split("\n")
                     if "experiment_" in l or "simple_trainer" in l]
            return [l.strip()[:120] for l in lines[:5]]
    except Exception:
        return []


def get_heartbeat_status():
    """Read all active heartbeat files."""
    heartbeats = {}
    if not os.path.isdir(HEARTBEAT_DIR):
        return heartbeats
    for fname in os.listdir(HEARTBEAT_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(HEARTBEAT_DIR, fname), encoding="utf-8") as f:
                    hb = json.load(f)
                heartbeats[fname.replace(".json", "")] = {
                    "status": hb.get("status"),
                    "timestamp": hb.get("timestamp"),
                    "dataset": hb.get("dataset"),
                    "run": hb.get("run"),
                    "gpu_mem_mb": hb.get("gpu_mem_mb"),
                }
            except Exception:
                pass
    return heartbeats


def _safe_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return -1


def tick():
    """Collect one snapshot."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    gpu = get_gpu_stats()
    heartbeats = get_heartbeat_status()

    entry = {
        "timestamp": now,
        "gpu": gpu,
        "training_heartbeats": heartbeats,
    }
    return entry


def run_monitor(interval=10, verbose=True):
    """Main monitoring loop."""
    os.makedirs(WATCHDOG_DIR, exist_ok=True)

    print(f"[Watchdog] Started. Interval={interval}s")
    print(f"[Watchdog] Log: {LOG_FILE}")
    print(f"[Watchdog] Latest: {LATEST_FILE}")

    tick_count = 0
    try:
        while True:
            entry = tick()
            tick_count += 1
            entry["tick"] = tick_count

            # Append to JSONL log (survives crash — each line is a complete record)
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())  # Force to disk immediately
            except Exception as e:
                print(f"[Watchdog] Log write error: {e}")

            # Overwrite latest snapshot
            try:
                with open(LATEST_FILE, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

            # Console output
            if verbose:
                gpu = entry.get("gpu", {})
                hbs = entry.get("training_heartbeats", {})
                hb_summary = ", ".join(
                    f"{k}={v.get('run','?')}" for k, v in hbs.items()
                    if v.get("status") == "running"
                ) or "none"
                print(
                    f"[{entry['timestamp']}] "
                    f"GPU: {gpu.get('gpu_util_pct','-')}% "
                    f"Mem: {gpu.get('mem_used_mb','-')}/{gpu.get('mem_total_mb','-')}MB "
                    f"Temp: {gpu.get('temp_c','-')}C "
                    f"Power: {gpu.get('power_w','-')}/{gpu.get('power_limit_w','-')}W "
                    f"| Training: {hb_summary}"
                )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n[Watchdog] Stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="GPU Watchdog Monitor")
    parser.add_argument("--interval", type=int, default=10,
                        help="Polling interval in seconds (default: 10)")
    parser.add_argument("--background", action="store_true",
                        help="Run silently (log only, no console output)")
    args = parser.parse_args()
    run_monitor(interval=args.interval, verbose=not args.background)


if __name__ == "__main__":
    main()
