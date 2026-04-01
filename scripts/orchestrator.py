#!/usr/bin/env python
r"""
VTBench Experiment Orchestrator
================================
Single-command automation for the full experiment suite.
Handles sequencing, parallel GPU execution, error recovery,
progress tracking, and resumability.

Usage:
    cd C:\Users\yangl\vtbench
    python scripts/orchestrator.py                    # run everything
    python scripts/orchestrator.py --resume           # skip completed experiments
    python scripts/orchestrator.py --phase B2         # start from specific phase
    python scripts/orchestrator.py --dry-run          # show plan without executing
    python scripts/orchestrator.py --solo             # no parallelism (safer, slower)

Hardware target: RTX 5070 Ti 16GB GDDR7, PyTorch 2.12+cu128
Strategy: 2 experiments in parallel (~3-5GB VRAM each), sequential phases
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# Configuration
# ============================================================

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", r"D:\chart_images")
RESULTS_DIR = "results"
STATE_FILE = os.path.join(RESULTS_DIR, "orchestrator_state.json")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

# Experiment definitions: (name, script, config, estimated_minutes)
# Order matters within phases; phases are sequential.
EXPERIMENTS = {
    # Phase A: CPU-only image generation
    "preflight": {
        "script": "scripts/preflight_check.py",
        "config": None,
        "gpu": False,
        "est_min": 60,
        "desc": "Audit & generate missing images",
    },

    # Phase B1: Warmup
    "6b": {
        "script": "scripts/experiment_6b_numerical_baseline.py",
        "config": "vtbench/config/experiment_6b_numerical.yaml",
        "gpu": True,
        "est_min": 30,
        "desc": "Numerical baselines (FCN, Transformer, OSCNN)",
    },

    # Phase B2
    "6a": {
        "script": "scripts/experiment_6a_encodings.py",
        "config": "vtbench/config/experiment_6a_encodings.yaml",
        "gpu": True,
        "est_min": 360,
        "desc": "Encoding comparison (~840 runs)",
    },
    "5i": {
        "script": "scripts/experiment_5i_backbones.py",
        "config": "vtbench/config/experiment_5i_backbones.yaml",
        "gpu": True,
        "est_min": 60,
        "desc": "Backbone comparison",
    },
    "5e": {
        "script": "scripts/experiment_5e_two_branch.py",
        "config": "vtbench/config/experiment_5e_full.yaml",
        "gpu": True,
        "est_min": 45,
        "desc": "Two-branch fusion",
    },
    "5b": {
        "script": "scripts/experiment_5b_multi_chart.py",
        "config": "vtbench/config/experiment_5b_full.yaml",
        "gpu": True,
        "est_min": 60,
        "desc": "Multi-chart fusion",
    },

    # Phase B3
    "7a": {
        "script": "scripts/experiment_7a_extended_encodings.py",
        "config": "vtbench/config/experiment_7a_extended.yaml",
        "gpu": True,
        "est_min": 120,
        "desc": "Extended encodings (phase space, diff, etc.)",
    },
    "5a": {
        "script": "scripts/experiment_5a_train_augmentation.py",
        "config": "vtbench/config/experiment_5a_full.yaml",
        "gpu": True,
        "est_min": 90,
        "desc": "Train augmentation (full)",
    },
    "5c": {
        "script": "scripts/experiment_5c_resnet18.py",
        "config": "vtbench/config/experiment_5c_full.yaml",
        "gpu": True,
        "est_min": 45,
        "desc": "ResNet18 (full)",
    },

    # Phase B4
    "7b": {
        "script": "scripts/experiment_7b_image_postprocess.py",
        "config": "vtbench/config/experiment_7b_postprocess.yaml",
        "gpu": True,
        "est_min": 180,
        "desc": "Image post-processing (~540 runs)",
    },
    "5d": {
        "script": "scripts/experiment_5d_rendering.py",
        "config": "vtbench/config/experiment_5d_full.yaml",
        "gpu": True,
        "est_min": 90,
        "desc": "Rendering variants (full)",
    },
    "5f": {
        "script": "scripts/experiment_5f_scale.py",
        "config": "vtbench/config/experiment_5f_scale.yaml",
        "gpu": True,
        "est_min": 60,
        "desc": "Dataset scale",
    },

    # Phase B5
    "7c": {
        "script": "scripts/experiment_7c_chart_ablation.py",
        "config": "vtbench/config/experiment_7c_chart_ablation.yaml",
        "gpu": True,
        "est_min": 150,
        "desc": "Chart ablation + augmentation (~420 runs)",
    },
    "5g": {
        "script": "scripts/experiment_5g_chart_type.py",
        "config": "vtbench/config/experiment_5g_chart_type.yaml",
        "gpu": True,
        "est_min": 45,
        "desc": "Chart type comparison",
    },
    "5h": {
        "script": "scripts/experiment_5h_resolution.py",
        "config": "vtbench/config/experiment_5h_resolution.yaml",
        "gpu": True,
        "est_min": 45,
        "desc": "Image resolution",
    },
    "5j": {
        "script": "scripts/experiment_5j_training.py",
        "config": "vtbench/config/experiment_5j_training.yaml",
        "gpu": True,
        "est_min": 90,
        "desc": "Training strategies",
    },

    # Phase B6
    "5k": {
        "script": "scripts/experiment_5k_ts_augment.py",
        "config": "vtbench/config/experiment_5k_ts_augment.yaml",
        "gpu": True,
        "est_min": 90,
        "desc": "TS augmentation",
    },
    "5m": {
        "script": "scripts/experiment_5m_transfer.py",
        "config": "vtbench/config/experiment_5m_transfer.yaml",
        "gpu": True,
        "est_min": 45,
        "desc": "Transfer learning",
    },

    # Phase B7
    "5l": {
        "script": "scripts/experiment_5l_ensemble.py",
        "config": "vtbench/config/experiment_5l_ensemble.yaml",
        "gpu": True,
        "est_min": 60,
        "desc": "Ensemble methods",
    },
    "8b": {
        "script": "scripts/experiment_8b_compute_profile.py",
        "config": "vtbench/config/experiment_8b_compute.yaml",
        "gpu": True,
        "est_min": 60,
        "desc": "Compute profiling",
    },

    # Phase B8: Headline
    "8a": {
        "script": "scripts/experiment_8a_broad_evaluation.py",
        "config": "vtbench/config/experiment_8a_broad.yaml",
        "gpu": True,
        "est_min": 480,
        "desc": "Broad dataset evaluation (~1800 runs) ** HEADLINE **",
    },

    # Phase C: Advanced methods
    "pregen_9a": {
        "script": "scripts/pregenerate_9a_encodings.py",
        "config": None,
        "gpu": False,
        "est_min": 20,
        "desc": "Pre-generate encoding images for 9A (CPU only)",
    },
    "9a": {
        "script": "scripts/experiment_9a_advanced.py",
        "config": "vtbench/config/experiment_9a_advanced.yaml",
        "gpu": True,
        "est_min": 600,
        "desc": "Advanced methods: new backbones, encodings, mixup, TTA, attention, ensemble",
    },
}

# Execution plan: CONSERVATIVE PARALLEL by VRAM profile.
#
# GPU power limit adjusted. Parallel ONLY for DeepCNN-only experiments
# (each ~200MB VRAM). Ensemble (3 models), 9A (4 models), 8A (long-running
# headline) always run solo to prevent driver corruption.
#
# VRAM budget for parallel: DeepCNN pair ≈ 400MB (safe on 16GB).
# NEVER parallel: 5L (3 models ~1.5GB), 9A (4 models ~1.5GB), 8A (headline).
#
# 5H deferred to cloud server (needs >16GB for 336×336).
PHASES = [
    {"name": "A",   "desc": "Image Generation (CPU)",          "solo_gpu": False, "slots": [["preflight"]]},
    {"name": "B1",  "desc": "Baselines",                       "solo_gpu": True,  "slots": [["6b"]]},
    {"name": "B2",  "desc": "6A Encoding comparison",          "solo_gpu": True,  "slots": [["6a"]]},
    {"name": "B3",  "desc": "Extended encodings",               "solo_gpu": True,  "slots": [["7a"]]},
    {"name": "B3b", "desc": "Image post-processing",            "solo_gpu": True,  "slots": [["7b"]]},
    # 5D + 5M: both DeepCNN-only, tiny VRAM (~200MB each) → safe parallel
    {"name": "B4",  "desc": "Rendering + Transfer (DeepCNN parallel)",
     "solo_gpu": False, "slots": [["5d"], ["5m"]]},
    # 5K: DeepCNN but augmented data up to 10x → solo to be safe
    {"name": "B4b", "desc": "TS Augmentation (up to 10x data)", "solo_gpu": True,  "slots": [["5k"]]},
    # 5L: 3 models (deepcnn+resnet18+efficientnet) → MUST be solo
    {"name": "B5",  "desc": "Ensemble (3 models, solo)",        "solo_gpu": True,  "slots": [["5l"]]},
    # 8B: 2 models but only 2 small datasets → solo (has resnet18)
    {"name": "B5b", "desc": "Compute profiling",                "solo_gpu": True,  "slots": [["8b"]]},
    # 8A: headline experiment, 17 datasets, long-running → solo
    {"name": "B6",  "desc": "8A Broad Evaluation (headline)",   "solo_gpu": True,  "slots": [["8a"]]},
    {"name": "C0",  "desc": "Pre-gen 9A images (CPU)",          "solo_gpu": False, "slots": [["pregen_9a"]]},
    # 9A: 4 models (deepcnn+resnet18+efficientnet+vit) → MUST be solo
    {"name": "C1",  "desc": "9A Advanced Methods (solo)",       "solo_gpu": True,  "slots": [["9a"]]},
]


# ============================================================
# Helpers
# ============================================================

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg, color=None):
    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "bold": "\033[1m",
    }
    reset = "\033[0m"
    prefix = f"[{ts()}]"
    if color and color in colors:
        print(f"{colors[color]}{prefix} {msg}{reset}")
    else:
        print(f"{prefix} {msg}")


def load_state():
    """Load orchestrator state (completed experiments)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"completed": {}, "failed": {}, "start_time": None}


def save_state(state):
    """Persist orchestrator state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def build_command(exp_key):
    """Build the command line for an experiment."""
    exp = EXPERIMENTS[exp_key]
    cmd = [sys.executable, exp["script"]]
    if exp["config"]:
        cmd.extend(["--config", exp["config"]])
    return cmd


def run_experiment(exp_key, log_path):
    """Run a single experiment, streaming output to log file and stdout."""
    exp = EXPERIMENTS[exp_key]
    cmd = build_command(exp_key)

    log(f"  START: {exp_key} - {exp['desc']}", "yellow")
    log(f"    cmd: {' '.join(cmd)}")
    log(f"    log: {log_path}")

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()

    try:
        env = os.environ.copy()
        env["CHART_IMAGE_ROOT"] = CHART_IMAGE_ROOT
        # Use PYTHONUNBUFFERED for real-time output
        env["PYTHONUNBUFFERED"] = "1"

        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                logf.write(line)
                logf.flush()
                # Print last-line summary to console (avoid flooding)
                stripped = line.strip()
                if stripped and (
                    stripped.startswith("[") or
                    "Accuracy" in stripped or
                    "ERROR" in stripped or
                    "Done" in stripped or
                    "complete" in stripped.lower()
                ):
                    print(f"    [{exp_key}] {stripped}")

            proc.wait()

        elapsed = time.time() - t0
        if proc.returncode == 0:
            log(f"  DONE: {exp_key} ({elapsed/60:.1f} min)", "green")
            return True, elapsed
        else:
            log(f"  FAILED: {exp_key} (exit code {proc.returncode}, {elapsed/60:.1f} min)", "red")
            return False, elapsed

    except Exception as e:
        elapsed = time.time() - t0
        log(f"  EXCEPTION: {exp_key} - {e}", "red")
        return False, elapsed


def run_slot(slot_exps, state, resume=False):
    """Run a sequential chain of experiments (one slot)."""
    results = {}
    for exp_key in slot_exps:
        # Skip if already completed (resume mode)
        if resume and exp_key in state.get("completed", {}):
            log(f"  SKIP (already done): {exp_key}", "cyan")
            results[exp_key] = (True, 0)
            continue

        log_path = os.path.join(LOG_DIR, f"{exp_key}.log")
        success, elapsed = run_experiment(exp_key, log_path)
        results[exp_key] = (success, elapsed)

        if success:
            state.setdefault("completed", {})[exp_key] = {
                "timestamp": ts(),
                "elapsed_min": round(elapsed / 60, 1),
            }
        else:
            state.setdefault("failed", {})[exp_key] = {
                "timestamp": ts(),
                "elapsed_min": round(elapsed / 60, 1),
            }
        save_state(state)

        # If a chain experiment fails, continue to next (don't block the chain)
        if not success:
            log(f"  WARNING: {exp_key} failed, continuing chain...", "yellow")

    return results


def run_slot_subprocess(slot_exps, state, resume=False):
    """
    Run a slot as a subprocess wrapper so we can do true parallelism.
    Returns a Popen object that runs a helper script.
    """
    # For parallel execution, we launch each slot as its own Python process
    # that runs experiments sequentially within the slot
    skip_list = []
    if resume:
        skip_list = [k for k in slot_exps if k in state.get("completed", {})]

    remaining = [k for k in slot_exps if k not in skip_list]
    if not remaining:
        return None, skip_list

    # Build a mini-script that runs the chain
    chain_cmds = []
    for exp_key in remaining:
        exp = EXPERIMENTS[exp_key]
        cmd_parts = [sys.executable, exp["script"]]
        if exp["config"]:
            cmd_parts.extend(["--config", exp["config"]])
        cmd_str = " ".join(cmd_parts)
        log_path = os.path.join(LOG_DIR, f"{exp_key}.log")
        chain_cmds.append((exp_key, cmd_str, log_path))

    # Create a temporary runner script
    runner_code = f"""
import subprocess, sys, os, time, json

os.environ["CHART_IMAGE_ROOT"] = {CHART_IMAGE_ROOT!r}
os.environ["PYTHONUNBUFFERED"] = "1"
state_file = {STATE_FILE!r}

def ts():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_state(key, success, elapsed):
    try:
        with open(state_file) as f:
            state = json.load(f)
    except:
        state = {{"completed": {{}}, "failed": {{}}}}

    bucket = "completed" if success else "failed"
    state.setdefault(bucket, {{}})[key] = {{
        "timestamp": ts(),
        "elapsed_min": round(elapsed / 60, 1),
    }}
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)

chain = {chain_cmds!r}
for exp_key, cmd, log_path in chain:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"[{{ts()}}] SLOT-START: {{exp_key}}")
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT,
                                  env=os.environ)
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        status = "OK" if ok else f"FAIL(rc={{proc.returncode}})"
        print(f"[{{ts()}}] SLOT-END: {{exp_key}} {{status}} ({{elapsed/60:.1f}} min)")
        update_state(exp_key, ok, elapsed)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"[{{ts()}}] SLOT-ERROR: {{exp_key}} {{e}}")
        update_state(exp_key, False, elapsed)
"""

    runner_path = os.path.join(LOG_DIR, f"_slot_{'_'.join(remaining)}.py")
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(runner_path, "w") as f:
        f.write(runner_code)

    env = os.environ.copy()
    env["CHART_IMAGE_ROOT"] = CHART_IMAGE_ROOT
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        [sys.executable, runner_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return proc, skip_list


# ============================================================
# Main Orchestrator
# ============================================================

def print_plan():
    """Print the full execution plan."""
    total_min = 0
    log("=" * 70, "cyan")
    log("VTBench Execution Plan", "bold")
    log("=" * 70, "cyan")

    for phase in PHASES:
        slot_strs = []
        phase_max = 0
        for slot in phase["slots"]:
            slot_time = sum(EXPERIMENTS[k]["est_min"] for k in slot)
            phase_max = max(phase_max, slot_time)
            exps = " -> ".join(f"{k}({EXPERIMENTS[k]['est_min']}m)" for k in slot)
            slot_strs.append(exps)

        total_min += phase_max
        parallel = " || " if len(phase["slots"]) > 1 else ""
        slot_display = f" || ".join(slot_strs)
        log(f"  Phase {phase['name']:4s} [{phase_max:3d}m] {phase['desc']}")
        log(f"         {slot_display}")

    log(f"\n  Estimated total wall time: {total_min}m ({total_min/60:.1f}h)", "cyan")
    log("=" * 70, "cyan")


def main():
    parser = argparse.ArgumentParser(description="VTBench Experiment Orchestrator")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed experiments")
    parser.add_argument("--phase", type=str, default=None,
                        help="Start from this phase (e.g., B2, B5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    parser.add_argument("--solo", action="store_true",
                        help="Run everything sequentially (no parallelism)")
    parser.add_argument("--only", nargs="*", default=None,
                        help="Run only these experiments (e.g., --only 6b 6a 8a)")
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    if args.dry_run:
        print_plan()
        state = load_state()
        done = list(state.get("completed", {}).keys())
        if done:
            log(f"\nAlready completed ({len(done)}): {', '.join(done)}", "green")
        return

    # ---- Single experiment mode ----
    if args.only:
        state = load_state()
        for exp_key in args.only:
            if exp_key not in EXPERIMENTS:
                log(f"Unknown experiment: {exp_key}", "red")
                continue
            if args.resume and exp_key in state.get("completed", {}):
                log(f"SKIP (done): {exp_key}", "cyan")
                continue
            log_path = os.path.join(LOG_DIR, f"{exp_key}.log")
            success, elapsed = run_experiment(exp_key, log_path)
            if success:
                state.setdefault("completed", {})[exp_key] = {
                    "timestamp": ts(), "elapsed_min": round(elapsed/60, 1)
                }
            else:
                state.setdefault("failed", {})[exp_key] = {
                    "timestamp": ts(), "elapsed_min": round(elapsed/60, 1)
                }
            save_state(state)
        return

    # ---- Full orchestration ----
    state = load_state()
    state["start_time"] = state.get("start_time") or ts()
    save_state(state)

    # Layer 3: Run crash report if previous run may have crashed
    if args.resume:
        log("Running post-crash diagnosis...", "yellow")
        try:
            subprocess.run(
                [sys.executable, "scripts/crash_report.py", "--brief"],
                timeout=10
            )
        except Exception:
            pass

    # Layer 2: Launch GPU watchdog as background process
    watchdog_proc = None
    watchdog_log_path = os.path.join(LOG_DIR, "watchdog_stdout.log")
    try:
        os.makedirs("results/watchdog", exist_ok=True)
        watchdog_logf = open(watchdog_log_path, "w", encoding="utf-8")
        watchdog_proc = subprocess.Popen(
            [sys.executable, "scripts/watchdog.py", "--interval", "10"],
            stdout=watchdog_logf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        log(f"GPU Watchdog started (PID={watchdog_proc.pid})", "green")
    except Exception as e:
        log(f"WARNING: Could not start watchdog: {e}", "yellow")

    # Verify environment
    log("=" * 70, "cyan")
    log("VTBench Orchestrator - RTX 5070 Ti 16GB", "bold")
    log("=" * 70, "cyan")

    log("Verifying PyTorch + CUDA...", "yellow")
    result = subprocess.run(
        [sys.executable, "-c",
         "import torch; v=torch.__version__; ok='OK' if torch.cuda.is_available() else 'FAIL'; "
         "print(f'PyTorch {v} CUDA: {ok}')"],
        capture_output=True, text=True
    )
    print(f"  {result.stdout.strip()}")
    if "FAIL" in result.stdout:
        log("FATAL: CUDA not available!", "red")
        sys.exit(1)

    print_plan()

    # Determine starting phase
    start_idx = 0
    if args.phase:
        for i, phase in enumerate(PHASES):
            if phase["name"].upper() == args.phase.upper():
                start_idx = i
                break
        else:
            log(f"Unknown phase: {args.phase}. Available: {[p['name'] for p in PHASES]}", "red")
            sys.exit(1)
        log(f"Starting from phase {args.phase}", "yellow")

    overall_t0 = time.time()

    for phase_idx, phase in enumerate(PHASES):
        if phase_idx < start_idx:
            log(f"Phase {phase['name']}: SKIPPED (--phase)", "cyan")
            continue

        log("", None)
        log(f"{'='*60}", "magenta")
        log(f"PHASE {phase['name']}: {phase['desc']}", "magenta")
        log(f"{'='*60}", "magenta")

        if args.solo or len(phase["slots"]) == 1:
            # Sequential execution
            for slot in phase["slots"]:
                run_slot(slot, state, resume=args.resume)
        else:
            # Parallel execution: launch each slot as a subprocess
            procs = []
            for slot in phase["slots"]:
                proc, skipped = run_slot_subprocess(slot, state, resume=args.resume)
                if skipped:
                    for k in skipped:
                        log(f"  SKIP (already done): {k}", "cyan")
                if proc:
                    procs.append((slot, proc))
                elif not skipped:
                    log(f"  Slot {slot}: nothing to run", "cyan")

            # Wait for all parallel slots
            if procs:
                log(f"  Waiting for {len(procs)} parallel slots...", "yellow")
                for slot, proc in procs:
                    for line in proc.stdout:
                        print(f"    {line.strip()}")
                    proc.wait()
                    rc = proc.returncode
                    status = "OK" if rc == 0 else f"FAIL(rc={rc})"
                    log(f"  Slot {slot}: {status}", "green" if rc == 0 else "red")

            # Reload state after parallel execution (subprocesses may have updated it)
            state = load_state()

        log(f"Phase {phase['name']} COMPLETE", "green")

    # ---- Summary ----
    total_elapsed = time.time() - overall_t0
    state = load_state()

    log("", None)
    log("=" * 70, "cyan")
    log("ORCHESTRATION COMPLETE", "bold")
    log("=" * 70, "cyan")
    log(f"Wall time: {total_elapsed/3600:.1f}h ({total_elapsed/60:.0f}m)", "cyan")

    completed = state.get("completed", {})
    failed = state.get("failed", {})

    log(f"Completed: {len(completed)}/{len(EXPERIMENTS)}", "green")
    if completed:
        total_compute = sum(v.get("elapsed_min", 0) for v in completed.values())
        log(f"Total compute time: {total_compute:.0f} min", "green")

    if failed:
        log(f"Failed: {len(failed)}", "red")
        for k, v in failed.items():
            log(f"  {k}: failed at {v.get('timestamp', '?')}", "red")
        log(f"\nTo re-run failed experiments:", "yellow")
        log(f"  python scripts/orchestrator.py --only {' '.join(failed.keys())}", "yellow")

    log(f"\nLogs: {LOG_DIR}/")
    log(f"State: {STATE_FILE}")
    log(f"Heartbeats: results/heartbeats/")
    log(f"GPU Monitor: results/watchdog/gpu_monitor.jsonl")

    # Stop watchdog
    if watchdog_proc:
        try:
            watchdog_proc.terminate()
            watchdog_proc.wait(timeout=5)
            log("GPU Watchdog stopped.", "green")
        except Exception:
            pass


if __name__ == "__main__":
    main()
