#!/usr/bin/env python
"""
VTBench Experiment Progress Checker
====================================
Comprehensive status report for all experiments.
Handles CSV deduplication (append-mode duplicates),
heartbeat analysis, and failure breakdown.

Usage:
    python scripts/check_progress.py
"""

import csv
import json
import os
import glob
import sys
from collections import Counter, defaultdict
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────
RESULTS_DIR = "results"
HEARTBEAT_DIR = os.path.join(RESULTS_DIR, "heartbeats")

# Define each experiment's CSV, dedup key columns, and expected total
EXPERIMENTS = {
    "4": {
        "csv": "experiment_4_smoke_test/accuracy_robustness.csv",
        "key": ["dataset", "chart_type", "model", "seed"],
        "expected": 48,
    },
    "5A": {
        "csv": "experiment_5a_full/augmentation_robustness_by_regime.csv",
        "key": ["dataset", "chart_type", "augmentation", "model", "seed"],
        "expected": 1200,
    },
    "5B": {
        "csv": "experiment_5b_full/accuracy_fusion.csv",
        "key": ["dataset", "fusion_mode", "model", "seed"],
        "expected": 150,
    },
    "5C": {
        "csv": "experiment_5c_full/accuracy_resnet18.csv",
        "key": ["dataset", "chart_type", "seed"],
        "expected": 150,
    },
    "5D": {
        "csv": "experiment_5d_full/accuracy_rendering_variants.csv",
        "key": ["dataset", "chart_type", "color_mode", "label_mode", "model", "seed"],
        "expected": 750,
    },
    "5E": {
        "csv": "experiment_5e_full/accuracy_two_branch.csv",
        "key": ["dataset", "chart_type", "numerical_model", "seed"],
        "expected": 300,
    },
    "5F": {
        "csv": "experiment_5f_scale/accuracy_results.csv",
        "key": ["dataset", "chart_type", "scale", "model", "seed"],
        "expected": 864,
    },
    "5G": {
        "csv": "experiment_5g_chart_type/accuracy_results.csv",
        "key": ["dataset", "chart_type", "model", "seed"],
        "expected": 384,
    },
    "5H": {
        "csv": "experiment_5h_resolution/accuracy_results.csv",
        "key": ["dataset", "chart_type", "resolution", "model", "seed"],
        "expected": 360,
    },
    "5I": {
        "csv": "experiment_5i_backbones/accuracy_results.csv",
        "key": ["dataset", "chart_type", "model", "seed"],
        "expected": 450,
    },
    "5J": {
        "csv": "experiment_5j_training/accuracy_results.csv",
        "key": ["dataset", "chart_type", "variant", "model", "seed"],
        "expected": 180,
    },
    "5K": {
        "csv": "experiment_5k_ts_augment/accuracy_results.csv",
        "key": ["dataset", "chart_type", "augmentation", "model", "seed"],
        "expected": 120,
    },
    "5L": {
        "csv": "experiment_5l_ensemble/accuracy_results.csv",
        "key": ["dataset", "ensemble_method", "seed"],
        "expected": 345,
    },
    "5M": {
        "csv": "experiment_5m_transfer/accuracy_results.csv",
        "key": ["dataset", "chart_type", "model", "seed"],
        "expected": 36,
    },
    "6A": {
        "csv": "experiment_6a_encodings/encoding_comparison.csv",
        "key": ["dataset", "encoding", "model", "seed"],
        "expected": 756,
    },
    "6B": {
        "csv": "experiment_6b_numerical/numerical_baselines.csv",
        "key": ["dataset", "model", "seed"],
        "expected": 63,
    },
    "7A": {
        "csv": "experiment_7a_extended/accuracy_extended_encodings.csv",
        "key": ["dataset", "encoding", "model", "seed"],
        "expected": 420,
    },
    "7B": {
        "csv": "experiment_7b_postprocess/accuracy_postprocess.csv",
        "key": ["dataset", "source", "postprocess", "model", "seed"],
        "expected": 540,
    },
    "7C": {
        "csv": "experiment_7c_chart_ablation/accuracy_chart_ablation.csv",
        "key": ["dataset", "chart_type", "label_mode", "augmentation", "model", "seed"],
        "expected": 390,
    },
    "8A": {
        "csv": "experiment_8a_broad_evaluation/broad_evaluation.csv",
        "key": ["dataset", "encoding", "model", "seed"],
        "expected": 1530,
    },
    "8B": {
        "csv": "experiment_8b_compute_profile/compute_profile.csv",
        "key": ["dataset", "encoding", "model"],
        "expected": 68,
    },
    "9A": {
        "csv": "experiment_9a_advanced/advanced_results.csv",
        "key": ["seed", "dataset", "encoding", "model", "method"],
        "expected": 1365,
    },
}


def safe_acc(row):
    """Extract accuracy from row, return -1 on any error."""
    try:
        return float(row.get("accuracy") or -1)
    except (ValueError, TypeError):
        return -1


def load_and_dedup(csv_path, key_cols_hint):
    """Load CSV, deduplicate by key (last occurrence wins).

    Auto-detects key columns: uses all columns except accuracy-like ones.
    Falls back to key_cols_hint if provided columns don't exist.
    """
    if not os.path.exists(csv_path):
        return None, 0, 0, 0, []

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    raw_total = len(rows)
    if raw_total == 0:
        return [], 0, 0, 0, rows

    # Auto-detect key columns: everything except accuracy/time/loss metrics
    skip_cols = {"accuracy", "train_time_s", "train_time", "time_s",
                 "loss", "val_loss", "val_acc", "test_acc",
                 "inference_time_ms", "gpu_peak_mb", "params",
                 "train_time_per_epoch", "total_train_time"}
    all_cols = list(rows[0].keys())
    auto_key = [c for c in all_cols if c.lower() not in skip_cols]

    # Use auto-detected key
    key_cols = auto_key if auto_key else key_cols_hint

    # Dedup: last occurrence per key wins
    seen = {}
    for r in rows:
        try:
            key = tuple(r.get(k, "") for k in key_cols)
        except Exception:
            continue
        seen[key] = r

    deduped = list(seen.values())
    ok = sum(1 for r in deduped if safe_acc(r) > 0)
    fail = sum(1 for r in deduped if safe_acc(r) <= 0)

    return deduped, raw_total, ok, fail, rows


def load_heartbeats():
    """Load all heartbeat JSON files."""
    hbs = {}
    if not os.path.isdir(HEARTBEAT_DIR):
        return hbs
    for f in glob.glob(os.path.join(HEARTBEAT_DIR, "*.json")):
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
            hbs[name] = data
        except Exception:
            pass
    return hbs


def format_duration(seconds):
    """Format seconds as human-readable duration."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def main():
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("=" * 80)
    print(f"  VTBench 实验进度报告  |  {now_str}")
    print("=" * 80)

    heartbeats = load_heartbeats()

    # ── Summary table ──
    print(f"\n{'实验':<6} {'去重后OK':>8} {'/ 去重总':>8} {'完成率':>7} "
          f"{'失败':>5} {'预期':>6} {'CSV行':>6}  状态")
    print("-" * 80)

    problem_exps = []  # Collect experiments with issues for detailed breakdown

    for exp_id, info in EXPERIMENTS.items():
        csv_path = os.path.join(RESULTS_DIR, info["csv"])
        key_cols = info["key"]
        expected = info["expected"]

        deduped, raw_total, ok, fail, raw_rows = load_and_dedup(csv_path, key_cols)

        if deduped is None:
            print(f"  {exp_id:<4}  {'CSV未找到':>55}  "
                  f"({os.path.basename(info['csv'])})")
            continue

        dedup_total = ok + fail
        pct = ok / dedup_total * 100 if dedup_total > 0 else 0

        # Status determination
        if pct >= 99:
            status = "✓ 完成"
        elif pct >= 90:
            status = "~ 基本完成"
        elif pct >= 50:
            status = "⚠ 部分完成"
        elif pct > 0:
            status = "✗ 大量失败"
        else:
            status = "✗✗ 全部失败"

        # Check heartbeat (try multiple key formats: "8a", "8A", etc.)
        hb_key = exp_id.lower().replace(" ", "")
        hb = heartbeats.get(hb_key) or heartbeats.get(exp_id)
        if hb and hb.get("status") == "running":
            ts = hb.get("timestamp", "")
            uptime = hb.get("uptime_sec", 0)
            status += f"  [运行中 {format_duration(uptime)}]"

        dup_note = ""
        if raw_total > dedup_total:
            dup_note = f" ({raw_total - dedup_total} 重复行)"

        print(f"  {exp_id:<4}  {ok:>7} / {dedup_total:<6} {pct:>6.1f}%  "
              f"{fail:>4}  {expected:>5}  {raw_total:>5}  {status}{dup_note}")

        if fail > 0 and pct < 95:
            problem_exps.append((exp_id, info, deduped, ok, fail))

    # ── Heartbeat details ──
    print("\n" + "=" * 80)
    print("  心跳文件状态")
    print("-" * 80)
    for name, hb in sorted(heartbeats.items()):
        status = hb.get("status", "?")
        ts = hb.get("timestamp", "?")
        pulse = hb.get("pulse_count", hb.get("total_pulses", "?"))
        uptime = hb.get("uptime_sec", hb.get("total_time_sec", 0))

        extra = ""
        if status == "running":
            ds = hb.get("dataset", "?")
            enc = hb.get("encoding", "?")
            model = hb.get("model", "?")
            method = hb.get("method", "")
            gpu = hb.get("gpu_mem_mb", 0)
            run_info = hb.get("run", "")
            extra = (f"\n         当前: {ds}/{enc}/{model}"
                     f"{('/' + method) if method else ''}"
                     f"  GPU={gpu:.0f}MB"
                     f"{('  进度=' + run_info) if run_info else ''}")

        print(f"  {name:<6} {status:<10} 脉冲={pulse:<5} "
              f"时长={format_duration(uptime):<8} 最后更新={ts}{extra}")

    # ── Failure breakdown for problem experiments ──
    if problem_exps:
        print("\n" + "=" * 80)
        print("  失败详情（去重后完成率 < 95% 的实验）")
        print("-" * 80)

        for exp_id, info, deduped, ok, fail in problem_exps:
            print(f"\n  ── {exp_id} ({ok}/{ok + fail} ok, {fail} 失败) ──")

            failed_rows = [r for r in deduped if safe_acc(r) <= 0]

            # Group by dataset
            by_ds = Counter(r.get("dataset", "?") for r in failed_rows)
            if by_ds:
                print("    按数据集:")
                for ds, cnt in by_ds.most_common(10):
                    print(f"      {ds}: {cnt} 失败")

            # Group by encoding (if applicable)
            enc_col = "encoding" if "encoding" in info["key"] else "source"
            if enc_col in info["key"]:
                by_enc = Counter(r.get(enc_col, "?") for r in failed_rows)
                if by_enc:
                    print("    按编码/来源:")
                    for enc, cnt in by_enc.most_common(10):
                        print(f"      {enc}: {cnt} 失败")

            # Group by model
            if "model" in info["key"]:
                by_model = Counter(r.get("model", "?") for r in failed_rows)
                if by_model:
                    print("    按模型:")
                    for m, cnt in by_model.most_common():
                        print(f"      {m}: {cnt} 失败")

            # Group by method (9A specific)
            if "method" in info["key"]:
                by_method = Counter(r.get("method", "?") for r in failed_rows)
                if by_method:
                    print("    按方法:")
                    for m, cnt in by_method.most_common():
                        print(f"      {m}: {cnt} 失败")

    # ── Quick accuracy summary for completed experiments ──
    print("\n" + "=" * 80)
    print("  已完成实验准确率排名（去重后，仅 OK runs）")
    print("-" * 80)

    acc_list = []
    for exp_id, info in EXPERIMENTS.items():
        csv_path = os.path.join(RESULTS_DIR, info["csv"])
        deduped, _, ok, fail, _ = load_and_dedup(csv_path, info["key"])
        if deduped is None or ok == 0:
            continue
        dedup_total = ok + fail
        pct = ok / dedup_total * 100 if dedup_total > 0 else 0
        if pct < 50:
            continue  # Skip mostly-failed experiments
        accs = [safe_acc(r) for r in deduped if safe_acc(r) > 0]
        if accs:
            avg = sum(accs) / len(accs)
            acc_list.append((exp_id, avg, min(accs), max(accs), len(accs)))

    acc_list.sort(key=lambda x: -x[1])
    for exp_id, avg, mn, mx, n in acc_list:
        bar = "█" * int(avg * 30) + "░" * (30 - int(avg * 30))
        print(f"  {exp_id:<4}  {bar}  {avg:.1%}  "
              f"(min={mn:.1%} max={mx:.1%} n={n})")

    print("\n" + "=" * 80)
    print("  完成。")
    print("=" * 80)


if __name__ == "__main__":
    main()
