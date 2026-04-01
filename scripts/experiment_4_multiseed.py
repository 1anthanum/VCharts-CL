"""
Experiment 4 (Multi-Seed): Augmentation Robustness with Repeated Trials
========================================================================
Wraps the core experiment_4 logic to run multiple seeds per
dataset/encoding combination, producing per-seed CSVs and an
aggregated summary with mean ± std.

Usage
-----
    python scripts/experiment_4_multiseed.py \
        --config vtbench/config/experiment_4_batch2.yaml
"""

import argparse
import copy
import csv
import gc
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure scripts/ is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vtbench.train.trainer import train_model
from vtbench.data.loader import create_dataloaders
from vtbench.data.chart_generator import GlobalYRangeCalculator
from vtbench.utils.gradcam import GradCAM, find_last_conv_layer
from vtbench.utils.augmentations import apply_augmentation, augmentation_label

# Re-use helpers from experiment_4
from experiment_4_augmentation_robustness import (
    _dataset_entries,
    _build_run_config,
    _ensure_base_images,
    _evaluate_accuracy,
    _evaluate_alignment_augmented,
    _gradcam_time_curve,
    _occlusion_curve,
    _max_shift_corr,
    _alignment_label,
    _save_comparison_figure,
)


def _set_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_indices(labels, preds, count, correct_only, seed):
    indices = np.arange(len(labels))
    if correct_only:
        indices = indices[preds == labels]
    indices = indices.tolist()
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    return indices[:count]


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ====================================================================
# Single-seed run  (returns accuracy_rows, alignment_rows)
# ====================================================================

def run_single_seed(cfg, seed, seed_idx, total_seeds):
    """Run the full experiment once with a specific seed.
    Returns (accuracy_rows, alignment_rows) — lists of dicts.
    """
    exp = cfg["experiment"]
    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    samples_per_dataset = int(exp.get("samples_per_dataset", 5))
    correct_only = bool(exp.get("correct_only", True))
    accuracy_only = bool(exp.get("accuracy_only", False))

    aug_specs = cfg.get("augmentations", [])
    occl_cfg = dict(cfg.get("occlusion", {}))
    output_dir = exp.get("output_dir", "results/experiment_4_multiseed")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Flatten augmentation specs
    aug_combos = []
    for spec in aug_specs:
        aug_type = spec["type"]
        for params in spec.get("levels", [{}]):
            lbl = augmentation_label(aug_type, params)
            aug_combos.append((aug_type, dict(params), lbl))

    accuracy_rows = []
    alignment_rows = []

    _set_seeds(seed)

    for ds_idx, dataset_entry in enumerate(datasets):
        dataset_name = dataset_entry["name"]
        print(f"\n{'=' * 60}")
        print(f"[Seed {seed_idx+1}/{total_seeds}, seed={seed}] "
              f"Dataset {ds_idx+1}/{len(datasets)}: {dataset_name}")
        print(f"{'=' * 60}")

        for chart_type in chart_types:
            print(f"\n  Encoding: {chart_type}")

            run_config = _build_run_config(cfg, dataset_entry, chart_type)
            _ensure_base_images(run_config)

            # Train model with this seed
            _set_seeds(seed)
            model = train_model(run_config).to(device)
            model.eval()

            loaders = create_dataloaders(run_config)
            test_loader = loaders["test"]["chart"]
            dataset_obj = test_loader.dataset

            # Clean accuracy
            clean_acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
            print(f"    Clean accuracy: {clean_acc:.4f}")

            accuracy_rows.append({
                "seed": seed,
                "dataset": dataset_name,
                "encoding": chart_type,
                "augmentation": "clean",
                "accuracy": clean_acc,
                "delta_accuracy": 0.0,
            })

            # Alignment setup (if needed)
            indices = []
            if not accuracy_only:
                preds_list, labels_list = [], []
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        logits = model(images)
                        pred = torch.argmax(logits, dim=1).cpu().numpy()
                        preds_list.extend(pred)
                        labels_list.extend(labels.numpy())
                preds_arr = np.array(preds_list)
                labels_arr = np.array(labels_list)
                indices = _select_indices(
                    labels_arr, preds_arr, samples_per_dataset, correct_only, seed
                )

                chart_cfg = run_config["chart_branches"]["branch_1"]
                global_y = dataset_obj.uni_global_y
                if global_y is None:
                    global_y = GlobalYRangeCalculator.calculate_global_y_range_univariate(
                        dataset_obj.time_series_data
                    )

                if indices:
                    tmp_dir = os.path.join(output_dir, "tmp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    clean_occl_cfg = dict(occl_cfg)
                    clean_occl_cfg["_tmp_dir"] = tmp_dir

                    target_layer = find_last_conv_layer(model)
                    gradcam = GradCAM(model, target_layer)

                    clean_align = _evaluate_alignment_augmented(
                        model, dataset_obj, indices, dataset_obj.transform, device,
                        chart_type, chart_cfg, global_y, clean_occl_cfg,
                        augment_fn=lambda img: img,
                        gradcam_obj=gradcam,
                    )
                    clean_corrs = [r["corr"] for r in clean_align]
                    clean_avg_corr = float(np.mean(clean_corrs)) if clean_corrs else 0.0
                    clean_avg_shift = float(np.mean([abs(r["shift"]) for r in clean_align])) if clean_align else 0.0

                    alignment_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "augmentation": "clean",
                        "avg_corr": clean_avg_corr,
                        "avg_abs_shift": clean_avg_shift,
                        "alignment_label": _alignment_label(clean_avg_corr),
                    })
                    gradcam.clear()

            # Augmentation sweep
            for aug_type, aug_params, aug_lbl in aug_combos:
                def _augment(img, _at=aug_type, _ap=aug_params):
                    return apply_augmentation(img, _at, _ap)

                aug_acc = _evaluate_accuracy(
                    model, test_loader, device, augment_fn=_augment
                )
                delta = aug_acc - clean_acc
                print(f"    {aug_lbl:>20s}: acc={aug_acc:.4f}  Δ={delta:+.4f}")

                accuracy_rows.append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "encoding": chart_type,
                    "augmentation": aug_lbl,
                    "accuracy": aug_acc,
                    "delta_accuracy": delta,
                })

                if not accuracy_only and indices:
                    target_layer_aug = find_last_conv_layer(model)
                    gradcam_aug = GradCAM(model, target_layer_aug)
                    aug_occl_cfg = dict(occl_cfg)
                    aug_occl_cfg["_tmp_dir"] = tmp_dir

                    aug_align = _evaluate_alignment_augmented(
                        model, dataset_obj, indices, dataset_obj.transform, device,
                        chart_type, chart_cfg, global_y, aug_occl_cfg,
                        augment_fn=_augment,
                        gradcam_obj=gradcam_aug,
                    )
                    aug_corrs = [r["corr"] for r in aug_align]
                    avg_corr = float(np.mean(aug_corrs)) if aug_corrs else 0.0
                    avg_shift = float(np.mean([abs(r["shift"]) for r in aug_align])) if aug_align else 0.0

                    alignment_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "augmentation": aug_lbl,
                        "avg_corr": avg_corr,
                        "avg_abs_shift": avg_shift,
                        "alignment_label": _alignment_label(avg_corr),
                    })
                    gradcam_aug.clear()
                    del aug_align

            # Memory cleanup
            del model, loaders, test_loader
            gc.collect()
            import resource
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"    [MEM] Peak RSS: {mem_mb:.0f} MB")

            # Incremental save after each encoding (crash-safe)
            _save_incremental(output_dir, accuracy_rows, alignment_rows, seed)

    return accuracy_rows, alignment_rows


def _save_incremental(output_dir, acc_rows, align_rows, seed):
    """Write partial results after each encoding to avoid data loss on crash."""
    inc_dir = os.path.join(output_dir, "incremental")
    os.makedirs(inc_dir, exist_ok=True)
    if acc_rows:
        path = os.path.join(inc_dir, f"accuracy_seed{seed}_partial.csv")
        keys = acc_rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(acc_rows)
    if align_rows:
        path = os.path.join(inc_dir, f"alignment_seed{seed}_partial.csv")
        keys = align_rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(align_rows)


# ====================================================================
# Aggregation: mean ± std across seeds
# ====================================================================

def aggregate_results(all_acc_rows, all_align_rows, output_dir):
    """Compute mean/std across seeds and write summary files."""

    # --- Accuracy ---
    acc_df = pd.DataFrame(all_acc_rows)
    acc_per_seed_path = os.path.join(output_dir, "accuracy_all_seeds.csv")
    acc_df.to_csv(acc_per_seed_path, index=False)
    print(f"  Saved per-seed accuracy: {acc_per_seed_path}")

    acc_agg = (
        acc_df
        .groupby(["dataset", "encoding", "augmentation"])
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            mean_delta=("delta_accuracy", "mean"),
            std_delta=("delta_accuracy", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )
    # Round for readability
    for col in ["mean_accuracy", "std_accuracy", "mean_delta", "std_delta"]:
        acc_agg[col] = acc_agg[col].round(4)

    acc_agg_path = os.path.join(output_dir, "accuracy_robustness_aggregated.csv")
    acc_agg.to_csv(acc_agg_path, index=False)
    print(f"  Saved aggregated accuracy: {acc_agg_path}")

    # --- Alignment ---
    if all_align_rows:
        align_df = pd.DataFrame(all_align_rows)
        align_per_seed_path = os.path.join(output_dir, "alignment_all_seeds.csv")
        align_df.to_csv(align_per_seed_path, index=False)
        print(f"  Saved per-seed alignment: {align_per_seed_path}")

        align_agg = (
            align_df
            .groupby(["dataset", "encoding", "augmentation"])
            .agg(
                mean_corr=("avg_corr", "mean"),
                std_corr=("avg_corr", "std"),
                mean_abs_shift=("avg_abs_shift", "mean"),
                std_abs_shift=("avg_abs_shift", "std"),
                n_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        for col in ["mean_corr", "std_corr", "mean_abs_shift", "std_abs_shift"]:
            align_agg[col] = align_agg[col].round(4)

        align_agg_path = os.path.join(output_dir, "alignment_robustness_aggregated.csv")
        align_agg.to_csv(align_agg_path, index=False)
        print(f"  Saved aggregated alignment: {align_agg_path}")
    else:
        align_agg = None

    # --- Summary markdown ---
    _write_aggregated_summary(output_dir, acc_agg, align_agg)

    return acc_agg, align_agg


def _write_aggregated_summary(output_dir, acc_agg, align_agg):
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Experiment 4 (Multi-Seed): Augmentation Robustness — Aggregated Summary\n\n")
        f.write(f"Seeds: {acc_agg['n_seeds'].iloc[0] if len(acc_agg) else '?'} repetitions\n\n")

        f.write("## Accuracy Robustness (mean ± std)\n\n")
        f.write("| Dataset | Encoding | Augmentation | Accuracy | ΔAccuracy |\n")
        f.write("|---------|----------|-------------|----------|------------|\n")
        for _, row in acc_agg.iterrows():
            f.write(
                f"| {row['dataset']} | {row['encoding']} | {row['augmentation']} "
                f"| {row['mean_accuracy']:.4f}±{row['std_accuracy']:.4f} "
                f"| {row['mean_delta']:+.4f}±{row['std_delta']:.4f} |\n"
            )

        if align_agg is not None and len(align_agg) > 0:
            f.write("\n## Alignment Robustness (mean ± std)\n\n")
            f.write("| Dataset | Encoding | Augmentation | Correlation | |Shift| |\n")
            f.write("|---------|----------|-------------|-------------|--------|\n")
            for _, row in align_agg.iterrows():
                f.write(
                    f"| {row['dataset']} | {row['encoding']} | {row['augmentation']} "
                    f"| {row['mean_corr']:.4f}±{row['std_corr']:.4f} "
                    f"| {row['mean_abs_shift']:.2f}±{row['std_abs_shift']:.2f} |\n"
                )

    print(f"  Saved summary: {md_path}")


# ====================================================================
# Main
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4 (Multi-Seed): Augmentation Robustness"
    )
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    cfg = load_config(args.config)

    exp = cfg["experiment"]
    seeds = exp.get("seeds", [42, 123, 7])
    output_dir = exp.get("output_dir", "results/experiment_4_multiseed")
    os.makedirs(output_dir, exist_ok=True)

    all_acc_rows = []
    all_align_rows = []

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'#' * 70}")
        print(f"# SEED RUN {seed_idx+1}/{len(seeds)} — seed={seed}")
        print(f"{'#' * 70}")

        acc_rows, align_rows = run_single_seed(cfg, seed, seed_idx, len(seeds))
        all_acc_rows.extend(acc_rows)
        all_align_rows.extend(align_rows)

        # Write intermediate per-seed CSV (in case of crash)
        _save_checkpoint(output_dir, all_acc_rows, all_align_rows, seed_idx + 1)

    print(f"\n{'#' * 70}")
    print(f"# AGGREGATING {len(seeds)} SEEDS")
    print(f"{'#' * 70}\n")

    aggregate_results(all_acc_rows, all_align_rows, output_dir)
    print("\nDone.")


def _save_checkpoint(output_dir, acc_rows, align_rows, seeds_done):
    """Write checkpoint CSV after each seed so we don't lose progress."""
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    acc_path = os.path.join(ckpt_dir, f"accuracy_after_{seeds_done}_seeds.csv")
    if acc_rows:
        keys = acc_rows[0].keys()
        with open(acc_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(acc_rows)

    if align_rows:
        align_path = os.path.join(ckpt_dir, f"alignment_after_{seeds_done}_seeds.csv")
        keys = align_rows[0].keys()
        with open(align_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(align_rows)

    print(f"  [Checkpoint] Saved after {seeds_done} seed(s) to {ckpt_dir}/")


if __name__ == "__main__":
    main()
