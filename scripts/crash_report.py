#!/usr/bin/env python
"""
Layer 3: Post-Crash Recovery Checker
=====================================
Run after a reboot/black screen to diagnose what happened.

Reads:
  - Watchdog JSONL log (Layer 2) — last GPU state before death
  - Training heartbeats (Layer 1) — last training state before death
  - Orchestrator state — which experiments completed/failed

Outputs a crash diagnosis report.

Usage:
    python scripts/crash_report.py              # full report
    python scripts/crash_report.py --brief      # one-line summary
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta


WATCHDOG_LOG = os.path.join("results", "watchdog", "gpu_monitor.jsonl")
HEARTBEAT_DIR = os.path.join("results", "heartbeats")
ORCHESTRATOR_STATE = os.path.join("results", "orchestrator_state.json")
REPORT_FILE = os.path.join("results", "watchdog", "crash_report.txt")


def load_last_watchdog_entries(n=10):
    """Load last N entries from watchdog JSONL."""
    if not os.path.exists(WATCHDOG_LOG):
        return []
    entries = []
    try:
        with open(WATCHDOG_LOG, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries[-n:]
    except Exception:
        return []


def load_heartbeats():
    """Load all heartbeat files."""
    heartbeats = {}
    if not os.path.isdir(HEARTBEAT_DIR):
        return heartbeats
    for fname in os.listdir(HEARTBEAT_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(HEARTBEAT_DIR, fname), "r",
                          encoding="utf-8") as f:
                    heartbeats[fname.replace(".json", "")] = json.load(f)
            except Exception:
                pass
    return heartbeats


def load_orchestrator_state():
    """Load orchestrator state."""
    if not os.path.exists(ORCHESTRATOR_STATE):
        return {}
    try:
        with open(ORCHESTRATOR_STATE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def classify_shutdown(last_watchdog, heartbeats):
    """
    Classify the type of shutdown:
      - normal: training completed, heartbeat says 'completed'
      - process_crash: watchdog alive but heartbeat stopped mid-run
      - hard_crash: watchdog also stopped abruptly (GPU/driver/system crash)
      - unknown: insufficient data
    """
    if not last_watchdog and not heartbeats:
        return "unknown", "No watchdog or heartbeat data found."

    # Check if any heartbeat is in a non-terminal state
    active_heartbeats = {
        k: v for k, v in heartbeats.items()
        if v.get("status") not in ("completed", "failed", "skipped")
    }

    if not active_heartbeats and not last_watchdog:
        return "normal", "All heartbeats show completed/failed. Clean shutdown."

    if not active_heartbeats:
        return "normal", "All experiments finished. Watchdog data present but no active runs."

    # We have active (non-completed) heartbeats = something was running when it died
    diagnosis_parts = []

    for name, hb in active_heartbeats.items():
        diagnosis_parts.append(
            f"  {name}: status={hb.get('status')}, "
            f"last_pulse={hb.get('timestamp')}, "
            f"dataset={hb.get('dataset')}, run={hb.get('run')}, "
            f"gpu_mem={hb.get('gpu_mem_mb')}MB"
        )

    if last_watchdog:
        last_entry = last_watchdog[-1]
        gpu = last_entry.get("gpu", {})

        # Check for warning signs in last watchdog entries
        high_mem = gpu.get("mem_used_mb", 0) > 14000  # >14GB of 16GB
        high_temp = gpu.get("temp_c", 0) > 85
        high_power = gpu.get("power_w", 0) > gpu.get("power_limit_w", 999) * 0.95

        risk_factors = []
        if high_mem:
            risk_factors.append(f"VRAM near limit: {gpu.get('mem_used_mb')}MB")
        if high_temp:
            risk_factors.append(f"High temp: {gpu.get('temp_c')}C")
        if high_power:
            risk_factors.append(
                f"Near power limit: {gpu.get('power_w')}W / {gpu.get('power_limit_w')}W")

        diagnosis_parts.append(f"\n  Last watchdog: {last_entry.get('timestamp')}")
        diagnosis_parts.append(
            f"  GPU state: util={gpu.get('gpu_util_pct')}%, "
            f"mem={gpu.get('mem_used_mb')}/{gpu.get('mem_total_mb')}MB, "
            f"temp={gpu.get('temp_c')}C, "
            f"power={gpu.get('power_w')}/{gpu.get('power_limit_w')}W"
        )

        if risk_factors:
            diagnosis_parts.append(f"  RISK FACTORS: {'; '.join(risk_factors)}")
            return "hard_crash", "\n".join(diagnosis_parts)
        else:
            # Check time gap between last watchdog and now
            try:
                last_ts = datetime.strptime(
                    last_entry["timestamp"], "%Y-%m-%d %H:%M:%S")
                gap = datetime.now() - last_ts
                if gap > timedelta(minutes=5):
                    diagnosis_parts.append(
                        f"  Time since last watchdog: {gap} (>5min = likely hard crash)")
                    return "hard_crash", "\n".join(diagnosis_parts)
            except Exception:
                pass

        return "hard_crash", "\n".join(diagnosis_parts)
    else:
        return "process_crash", (
            "Watchdog had no data, but heartbeats were active:\n"
            + "\n".join(diagnosis_parts)
        )


def generate_report(brief=False):
    """Generate crash diagnosis report."""
    watchdog_entries = load_last_watchdog_entries(20)
    heartbeats = load_heartbeats()
    orch_state = load_orchestrator_state()

    crash_type, diagnosis = classify_shutdown(watchdog_entries, heartbeats)

    if brief:
        print(f"[Crash Report] Type: {crash_type}")
        if crash_type != "normal":
            print(f"  {diagnosis.split(chr(10))[0]}")
        return

    # Full report
    lines = []
    lines.append("=" * 70)
    lines.append("VTBench Crash Diagnosis Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"CLASSIFICATION: {crash_type.upper()}")
    lines.append("")
    lines.append("DIAGNOSIS:")
    lines.append(diagnosis)
    lines.append("")

    # Orchestrator state
    completed = list(orch_state.get("completed", {}).keys())
    failed = list(orch_state.get("failed", {}).keys())
    lines.append(f"ORCHESTRATOR: {len(completed)} completed, {len(failed)} failed")
    lines.append(f"  Completed: {', '.join(sorted(completed))}")
    if failed:
        lines.append(f"  Failed: {', '.join(sorted(failed))}")

    # Last 5 watchdog entries (GPU trend before crash)
    if watchdog_entries:
        lines.append("")
        lines.append("GPU TREND (last entries before crash):")
        for entry in watchdog_entries[-5:]:
            gpu = entry.get("gpu", {})
            lines.append(
                f"  [{entry.get('timestamp')}] "
                f"util={gpu.get('gpu_util_pct', '?')}% "
                f"mem={gpu.get('mem_used_mb', '?')}MB "
                f"temp={gpu.get('temp_c', '?')}C "
                f"power={gpu.get('power_w', '?')}W"
            )

    # Heartbeat details
    if heartbeats:
        lines.append("")
        lines.append("HEARTBEAT STATUS:")
        for name, hb in sorted(heartbeats.items()):
            lines.append(
                f"  {name}: status={hb.get('status')}, "
                f"time={hb.get('timestamp')}, "
                f"dataset={hb.get('dataset', '?')}, "
                f"run={hb.get('run', '?')}"
            )

    # Recovery suggestions
    lines.append("")
    lines.append("RECOVERY:")
    if crash_type == "hard_crash":
        lines.append("  1. Check GPU driver version and stability")
        lines.append("  2. Review power limit settings")
        lines.append("  3. Run: python scripts/orchestrator.py --resume")
        lines.append("  4. Experiments with CSV resume will pick up where they left off")
    elif crash_type == "process_crash":
        lines.append("  1. Check experiment logs in results/logs/")
        lines.append("  2. Run: python scripts/orchestrator.py --resume")
    else:
        lines.append("  No action needed — clean shutdown detected.")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    # Save to file
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    try:
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nReport saved to: {REPORT_FILE}")
    except Exception as e:
        print(f"\nFailed to save report: {e}")


def main():
    parser = argparse.ArgumentParser(description="Post-Crash Recovery Checker")
    parser.add_argument("--brief", action="store_true",
                        help="One-line summary only")
    args = parser.parse_args()
    generate_report(brief=args.brief)


if __name__ == "__main__":
    main()
