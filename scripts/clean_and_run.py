#!/usr/bin/env python
"""
VTBench: Clean CSVs + Generate missing encodings + Run experiments 7B → 8A → 9A
================================================================================

Usage (PowerShell):

    # 推荐: 一条命令全部搞定 (CSV清理 + 编码生成 + 顺序训练)
    python scripts/clean_and_run.py

    # 仅清理CSV + 生成编码 (不运行训练)
    python scripts/clean_and_run.py --prep-only

    # 仅运行指定实验 (跳过清理和生成)
    python scripts/clean_and_run.py --run-only 7B
    python scripts/clean_and_run.py --run-only 8A
    python scripts/clean_and_run.py --run-only 9A

Steps:
  1. Clean 7B/8A/9A CSVs (remove accuracy=-1 rows, backup originals)
  2. Generate missing encoding images for 9A (wavelet_scattering, signature, persistence)
  3. Run 7B → 8A → 9A sequentially (single GPU, no contention)

重要: 请确保运行前没有其他 GPU 实验在运行！
     用 tasklist | findstr python 检查是否有其他 Python 进程占用 GPU。
"""

import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ============================================================
# Step 1: Clean CSVs (remove accuracy=-1 rows)
# ============================================================
def clean_csv(csv_path):
    """Remove rows with accuracy <= 0 from CSV, keep header + successful rows."""
    if not os.path.exists(csv_path):
        log(f"  跳过 (文件不存在): {csv_path}")
        return 0, 0

    # Backup original
    backup = csv_path + f".backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(csv_path, backup)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    kept = []
    removed = 0
    for r in rows:
        try:
            acc = float(r.get("accuracy") or -1)
        except (ValueError, TypeError):
            acc = -1
        if acc > 0:
            kept.append(r)
        else:
            removed += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    log(f"  {csv_path}")
    log(f"    原始: {total} 行 → 保留: {len(kept)} 行, 删除: {removed} 行")
    log(f"    备份: {backup}")
    return len(kept), removed


def step1_clean():
    log("=" * 60)
    log("步骤 1: 清理 CSV (删除 accuracy=-1 的失败行)")
    log("=" * 60)

    csvs = [
        "results/experiment_7b_postprocess/accuracy_postprocess.csv",
        "results/experiment_8a_broad_evaluation/broad_evaluation.csv",
        "results/experiment_9a_advanced/advanced_results.csv",
    ]

    total_kept = 0
    total_removed = 0
    for p in csvs:
        k, r = clean_csv(p)
        total_kept += k
        total_removed += r

    log(f"\n  总计: 保留 {total_kept} 行, 删除 {total_removed} 行")


# ============================================================
# Step 2: Generate missing encoding images for 9A
# ============================================================
def step2_generate_encodings():
    log("\n" + "=" * 60)
    log("步骤 2: 为 9A 生成缺失的编码图像")
    log("=" * 60)

    datasets = ["SyntheticControl", "GunPoint", "CBF", "Trace", "ECG5000"]
    encodings = ["wavelet_scattering", "signature", "persistence"]
    chart_root = os.environ.get("CHART_IMAGE_ROOT", "chart_images")

    # Check which need generating
    missing = []
    for ds in datasets:
        for enc in encodings:
            enc_dir = os.path.join(chart_root, f"{ds}_images", enc)
            if not os.path.isdir(enc_dir):
                missing.append((ds, enc))
            else:
                train_dir = os.path.join(enc_dir, "train")
                test_dir = os.path.join(enc_dir, "test")
                if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
                    missing.append((ds, enc))

    if not missing:
        log("  所有编码图像已存在，跳过生成。")
        return

    log(f"  需要生成 {len(missing)} 个 数据集×编码 组合:")
    for ds, enc in missing:
        log(f"    {ds} / {enc}")

    # Generate using preflight_check.py or inline generation
    log("\n  开始生成...")

    # Use inline generation for reliability
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from vtbench.data.loader import read_ucr
    from vtbench.data.ts_image_encodings import get_encoding
    from PIL import Image
    import numpy as np

    for ds, enc in missing:
        log(f"\n  生成 {ds}/{enc} ...")

        # Find dataset files
        train_path = test_path = None
        for ext in [".tsv", ".ts"]:
            tp = f"UCRArchive_2018/{ds}/{ds}_TRAIN{ext}"
            if os.path.exists(tp):
                train_path = tp
                test_path = f"UCRArchive_2018/{ds}/{ds}_TEST{ext}"
                break

        if train_path is None:
            log(f"    跳过: 数据集文件未找到")
            continue

        X_train, y_train = read_ucr(train_path)
        X_test, y_test = read_ucr(test_path)

        base_dir = os.path.join(chart_root, f"{ds}_images", enc)
        train_dir = os.path.join(base_dir, "train")
        test_dir = os.path.join(base_dir, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        n_train = len(X_train)

        # Generate train images
        for i, ts in enumerate(X_train):
            img_arr = get_encoding(enc, ts, image_size=128)
            if img_arr.ndim == 2:
                img = Image.fromarray(img_arr.astype(np.uint8), mode='L').convert('RGB')
            elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
                img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')
            else:
                img = Image.fromarray(img_arr.astype(np.uint8))
            img.save(os.path.join(train_dir, f"sample_{i}.png"))

        # Generate test images (global indices starting from n_train)
        for i, ts in enumerate(X_test):
            img_arr = get_encoding(enc, ts, image_size=128)
            if img_arr.ndim == 2:
                img = Image.fromarray(img_arr.astype(np.uint8), mode='L').convert('RGB')
            elif img_arr.ndim == 3 and img_arr.shape[2] == 3:
                img = Image.fromarray(img_arr.astype(np.uint8), mode='RGB')
            else:
                img = Image.fromarray(img_arr.astype(np.uint8))
            img.save(os.path.join(test_dir, f"sample_{n_train + i}.png"))

        log(f"    完成: {n_train} train + {len(X_test)} test 图像")

    log("\n  编码图像生成完毕。")


# ============================================================
# Step 3: Run experiments sequentially
# ============================================================
def run_experiment(name, script, config):
    """Run a single experiment, return success/failure."""
    log(f"\n{'=' * 60}")
    log(f"运行实验 {name}")
    log(f"  脚本: {script}")
    log(f"  配置: {config}")
    log(f"{'=' * 60}")

    if not os.path.exists(script):
        log(f"  错误: 脚本不存在 {script}")
        return False
    if not os.path.exists(config):
        log(f"  错误: 配置不存在 {config}")
        return False

    t0 = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script, "--config", config],
            check=False,
            timeout=3600 * 10,  # 10 hour timeout per experiment
        )
        elapsed = time.time() - t0
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)

        if result.returncode == 0:
            log(f"\n  ✓ {name} 完成 (耗时 {hours}h{mins:02d}m)")
            return True
        else:
            log(f"\n  ✗ {name} 退出码 {result.returncode} (耗时 {hours}h{mins:02d}m)")
            return False

    except subprocess.TimeoutExpired:
        log(f"\n  ✗ {name} 超时 (>10小时)")
        return False
    except Exception as e:
        log(f"\n  ✗ {name} 异常: {e}")
        return False


def step3_run():
    log("\n" + "=" * 60)
    log("步骤 3: 顺序运行实验 (7B → 8A → 9A)")
    log("  重要: 请确保没有其他 GPU 实验在运行！")
    log("=" * 60)

    experiments = [
        ("7B", "scripts/experiment_7b_image_postprocess.py",
         "vtbench/config/experiment_7b_postprocess.yaml"),
        ("8A", "scripts/experiment_8a_broad_evaluation.py",
         "vtbench/config/experiment_8a_broad.yaml"),
        ("9A", "scripts/experiment_9a_advanced.py",
         "vtbench/config/experiment_9a_advanced.yaml"),
    ]

    results = {}
    for name, script, config in experiments:
        success = run_experiment(name, script, config)
        results[name] = success

        if not success:
            log(f"\n  警告: {name} 失败，继续下一个实验...")

    # Summary
    log("\n" + "=" * 60)
    log("运行总结")
    log("=" * 60)
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败"
        log(f"  {name}: {status}")

    log(f"\n运行进度检查: python scripts/check_progress.py")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VTBench 实验重跑流程")
    parser.add_argument("--prep-only", action="store_true",
                        help="仅清理CSV + 生成编码，不运行训练")
    parser.add_argument("--run-only", type=str, default=None,
                        choices=["7B", "8A", "9A", "7b", "8a", "9a"],
                        help="仅运行指定实验（跳过清理和生成）")
    args = parser.parse_args()

    log("=" * 60)
    log("VTBench 实验重跑流程")
    log(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 60)

    if args.run_only:
        # Run single experiment directly
        exp = args.run_only.upper()
        exp_map = {
            "7B": ("7B", "scripts/experiment_7b_image_postprocess.py",
                   "vtbench/config/experiment_7b_postprocess.yaml"),
            "8A": ("8A", "scripts/experiment_8a_broad_evaluation.py",
                   "vtbench/config/experiment_8a_broad.yaml"),
            "9A": ("9A", "scripts/experiment_9a_advanced.py",
                   "vtbench/config/experiment_9a_advanced.yaml"),
        }
        if exp in exp_map:
            name, script, config = exp_map[exp]
            run_experiment(name, script, config)
    else:
        # Step 1: Clean CSVs
        step1_clean()

        # Step 2: Generate missing encodings
        step2_generate_encodings()

        if args.prep_only:
            log("\n--prep-only 模式: 清理和生成完毕，跳过训练。")
            log("接下来可以运行:")
            log("  python scripts/clean_and_run.py --run-only 7B")
            log("  python scripts/clean_and_run.py --run-only 8A")
            log("  python scripts/clean_and_run.py --run-only 9A")
        else:
            # Step 3: Run experiments
            step3_run()

    log(f"\n流程结束: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
