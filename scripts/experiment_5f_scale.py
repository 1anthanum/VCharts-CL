#!/usr/bin/env python
"""
Experiment 5F: Dataset Scale Effect
====================================
Tests chart-based classification across datasets of varying training set
sizes (20 to 3601 samples) to determine whether sample size is the primary
driver of accuracy. Compares DeepCNN (random init) vs ResNet18 (pretrained).

Usage
-----
    python scripts/experiment_5f_scale.py \
        --config vtbench/config/experiment_5f_scale.yaml
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
    output_dir = exp.get("output_dir", "results/experiment_5f_scale")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])
    model_configs = exp.get("model_configs", [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False},
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5f_scale", config=cfg)

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

            for mc in model_configs:
                model_name = mc["name"]

                for chart_type in chart_types:
                    print(f"\n{'=' * 60}")
                    print(f"Dataset: {dataset_name} (seed={seed})")
                    print(f"  Model: {model_name}  Chart: {chart_type}")
                    print(f"{'=' * 60}")

                    # Build config with model override
                    run_config = _build_run_config(cfg, dataset_entry, chart_type)
                    run_config["model"]["chart_model"] = mc["chart_model"]
                    run_config["model"]["pretrained"] = mc["pretrained"]

                    # Adjust LR for ResNet18
                    if mc["chart_model"] == "resnet18":
                        run_config["training"]["learning_rate"] = 0.0005

                    _ensure_base_images(run_config)

                    # Train
                    _set_seeds(seed)
                    try:
                        model = train_model(run_config).to(device)
                    except Exception as e:
                        print(f"    TRAINING FAILED: {e}")
                        accuracy_rows.append({
                            "seed": seed,
                            "dataset": dataset_name,
                            "model": model_name,
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
                    print(f"      Clean accuracy: {clean_acc:.4f}")
                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "model": model_name,
                        "chart_type": chart_type,
                        "augmentation": "clean",
                        "accuracy": clean_acc,
                    })

                    # Log to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}_{model_name}_{chart_type}_seed{seed}",
                        config={
                            "dataset": dataset_name,
                            "model": model_name,
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
                            "model": model_name,
                            "chart_type": chart_type,
                            "augmentation": aug_lbl,
                            "accuracy": aug_acc,
                        })

                    # Free memory
                    del model, loaders, test_loader
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    fieldnames = ["seed", "dataset", "model", "chart_type", "augmentation", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nResults saved to {csv_path}")

    # Print summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY — Mean Clean Accuracy by Dataset x Model x ChartType")
    print(f"{'=' * 70}")
    from collections import defaultdict
    acc_map = defaultdict(list)
    for row in accuracy_rows:
        if row["augmentation"] == "clean" and row["accuracy"] >= 0:
            key = (row["dataset"], row["model"], row["chart_type"])
            acc_map[key].append(row["accuracy"])

    prev_ds = None
    for (ds, mdl, ct), accs in sorted(acc_map.items()):
        if ds != prev_ds:
            print(f"\n  {ds}:")
            prev_ds = ds
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"    {mdl:>22s} / {ct:<8s}:  {mean_acc:.4f} +/- {std_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5F: Dataset Scale Effect"
    )
    parser.add_argument(
        "--config", required=True, help="YAML config file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
