#!/usr/bin/env python
"""
Experiment 5C: ResNet18 Pretrained
===================================
Evaluates the effectiveness of pretrained ResNet18 as a chart encoder
compared to non-pretrained and other architectures (DeepCNN, SimpleCNN).

Tests ResNet18 with pretrained weights across chart types and datasets,
measuring robustness to test-time augmentations.

Usage
-----
    python scripts/experiment_5c_resnet18.py \
        --config vtbench/config/experiment_5c.yaml
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
from PIL import Image

# Setup path to import experiment_4 helpers
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


# ====================================================================
# Main experiment loop
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5c_resnet18")
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="5c_resnet18", config=cfg)

    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Result collectors
    accuracy_rows = []

    # Flatten augmentation specs into (type, params, label) triples
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

                # Train model with ResNet18 (pretrained=true)
                _set_seeds(seed)
                model = train_model(run_config).to(device)
                model.eval()

                loaders = create_dataloaders(run_config)
                test_loader = loaders["test"]["chart"]

                # Compute clean baseline accuracy
                clean_acc = _evaluate_accuracy(
                    model, test_loader, device, augment_fn=None
                )
                print(f"    Clean accuracy: {clean_acc:.4f}")

                accuracy_rows.append({
                    "seed": seed,
                    "dataset": dataset_name,
                    "encoding": chart_type,
                    "model_name": "resnet18",
                    "augmentation": "clean",
                    "accuracy": f"{clean_acc:.4f}",
                    "delta_accuracy": "0.0000",
                })

                wb.log_run_result(
                    name=f"{dataset_name}/{chart_type}/s{seed}",
                    config={"dataset": dataset_name, "encoding": chart_type, "model": "resnet18", "seed": seed},
                    accuracy=clean_acc,
                )

                # Augmentation sweep
                for aug_type, aug_params, aug_lbl in aug_combos:
                    def _augment(img, _at=aug_type, _ap=aug_params):
                        return apply_augmentation(img, _at, _ap)

                    aug_acc = _evaluate_accuracy(
                        model, test_loader, device, augment_fn=_augment
                    )
                    delta = aug_acc - clean_acc
                    print(f"    {aug_lbl:>20s}: acc={aug_acc:.4f}  d={delta:+.4f}")

                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "model_name": "resnet18",
                        "augmentation": aug_lbl,
                        "accuracy": f"{aug_acc:.4f}",
                        "delta_accuracy": f"{delta:.4f}",
                    })

                # Memory cleanup
                del model, loaders, test_loader
                gc.collect()

    # ================================================================
    # Write summary CSV
    # ================================================================

    acc_csv = os.path.join(output_dir, "accuracy_resnet18.csv")
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "dataset", "encoding", "model_name",
                "augmentation", "accuracy", "delta_accuracy"
            ],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nWrote results to {acc_csv}")

    # Write summary markdown
    _write_summary_md(output_dir, accuracy_rows)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - accuracy_resnet18.csv")
    print(f"  - summary.md")


def _write_summary_md(output_dir, accuracy_rows):
    """Generate a markdown summary for quick inspection."""
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 5C: ResNet18 Pretrained — Summary\n\n")

        f.write("## Accuracy (Clean + Augmented)\n\n")
        f.write(
            "| Seed | Dataset | Encoding | Augmentation | Accuracy | dAccuracy |\n"
        )
        f.write("|------|---------|----------|--------------|----------|----------|\n")
        for row in accuracy_rows:
            f.write(
                f"| {row['seed']} | {row['dataset']} | {row['encoding']} | "
                f"{row['augmentation']} | {row['accuracy']} | {row['delta_accuracy']} |\n"
            )


# ====================================================================
# Config loading and CLI entry point
# ====================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5C: ResNet18 Pretrained"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
