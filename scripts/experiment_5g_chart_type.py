#!/usr/bin/env python
"""
Experiment 5G: Chart Type Systematic Comparison
=================================================
Head-to-head comparison of all 4 chart types (line, area, bar, scatter)
across multiple datasets. Motivated by 5A finding: Trace scatter=1.0
vs line=0.19, suggesting chart type selection is dataset-dependent.

Usage
-----
    python scripts/experiment_5g_chart_type.py \
        --config vtbench/config/experiment_5g_chart_type.yaml
"""

import argparse
import copy
import csv
import gc
import os
import random
import sys

import numpy as np
import torch
import yaml

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from vtbench.train.trainer import train_model
from vtbench.data.loader import create_dataloaders
from vtbench.utils.augmentations import apply_augmentation, augmentation_label
from vtbench.utils.wandb_logger import WandbLogger
from vtbench.utils.experiment_helpers import (
    _dataset_entries,
    _build_run_config,
    _ensure_base_images,
    _evaluate_accuracy,
    _set_seeds,
)


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5g_chart_type")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5g_chart_type", config=cfg)

    accuracy_rows = []

    # Flatten augmentation specs
    aug_specs = cfg.get("augmentations", [])
    aug_combos = []
    for spec in aug_specs:
        aug_type = spec["type"]
        for params in spec.get("levels", [{}]):
            lbl = augmentation_label(aug_type, params)
            aug_combos.append((aug_type, dict(params), lbl))

    for seed in seeds:
        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name} (seed={seed})")
            print(f"{'=' * 60}")

            for chart_type in chart_types:
                print(f"\n  Encoding: {chart_type}")

                run_config = _build_run_config(cfg, dataset_entry, chart_type)
                _ensure_base_images(run_config)

                _set_seeds(seed)
                try:
                    model = train_model(run_config).to(device)
                except Exception as e:
                    print(f"    TRAINING FAILED: {e}")
                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "chart_type": chart_type,
                        "augmentation": "clean",
                        "accuracy": -1,
                    })
                    continue

                model.eval()
                loaders = create_dataloaders(run_config)
                test_loader = loaders["test"]["chart"]

                # Clean accuracy
                clean_acc = _evaluate_accuracy(
                    model, test_loader, device, augment_fn=None
                )
                print(f"    Clean accuracy: {clean_acc:.4f}")
                accuracy_rows.append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "chart_type": chart_type,
                    "augmentation": "clean",
                    "accuracy": clean_acc,
                })

                # Log to W&B
                wb.log_run_result(
                    name=f"{dataset_name}_{chart_type}_seed{seed}",
                    config={
                        "dataset": dataset_name,
                        "chart_type": chart_type,
                        "seed": seed,
                    },
                    accuracy=clean_acc,
                )

                # Augmented accuracy
                for aug_type, aug_params, aug_lbl in aug_combos:
                    def augment_fn(img, _at=aug_type, _ap=aug_params):
                        return apply_augmentation(img, _at, _ap)

                    aug_acc = _evaluate_accuracy(
                        model, test_loader, device, augment_fn=augment_fn
                    )
                    d = aug_acc - clean_acc
                    print(f"      {aug_lbl}: acc={aug_acc:.4f}  d={d:+.4f}")
                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "chart_type": chart_type,
                        "augmentation": aug_lbl,
                        "accuracy": aug_acc,
                    })

                del model, loaders, test_loader
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    fieldnames = ["seed", "dataset", "chart_type", "augmentation", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nResults saved to {csv_path}")

    # Print summary: mean clean accuracy per dataset x chart_type
    print(f"\n{'=' * 60}")
    print("SUMMARY — Mean Clean Accuracy by Dataset x ChartType")
    print(f"{'=' * 60}")
    from collections import defaultdict
    acc_map = defaultdict(list)
    for row in accuracy_rows:
        if row["augmentation"] == "clean" and row["accuracy"] >= 0:
            key = (row["dataset"], row["chart_type"])
            acc_map[key].append(row["accuracy"])

    # Print as table
    all_ct = sorted(set(ct for (_, ct) in acc_map.keys()))
    header = f"{'Dataset':<22s}" + "".join(f"{ct:>10s}" for ct in all_ct)
    print(header)
    print("-" * len(header))

    all_ds = sorted(set(ds for (ds, _) in acc_map.keys()))
    for ds in all_ds:
        row_str = f"{ds:<22s}"
        for ct in all_ct:
            accs = acc_map.get((ds, ct), [])
            if accs:
                row_str += f"{np.mean(accs):>10.4f}"
            else:
                row_str += f"{'N/A':>10s}"
        print(row_str)

    # Best chart type per dataset
    print(f"\nBest chart type per dataset:")
    for ds in all_ds:
        best_ct, best_acc = None, -1
        for ct in all_ct:
            accs = acc_map.get((ds, ct), [])
            if accs and np.mean(accs) > best_acc:
                best_acc = np.mean(accs)
                best_ct = ct
        print(f"  {ds}: {best_ct} ({best_acc:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5G: Chart Type Comparison"
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
