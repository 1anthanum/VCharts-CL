#!/usr/bin/env python
"""
Experiment 5I: Modern Backbone Comparison
==========================================
Compares DeepCNN, ResNet18 (pretrained), EfficientNet-B0 (pretrained),
and ViT-Tiny across datasets and chart types.

Usage
-----
    python scripts/experiment_5i_backbones.py \
        --config vtbench/config/experiment_5i_backbones.yaml
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
from vtbench.train.factory import get_chart_model
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
    output_dir = exp.get("output_dir", "results/experiment_5i_backbones")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])
    model_configs = exp.get("model_configs", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    accuracy_rows = []

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5i_backbones", config=cfg)

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
                model_lr = mc.get("lr", 0.001)

                for chart_type in chart_types:
                    print(f"\n{'=' * 60}")
                    print(f"Dataset: {dataset_name} (seed={seed})")
                    print(f"  Model: {model_name}  Chart: {chart_type}")
                    print(f"{'=' * 60}")

                    run_config = _build_run_config(cfg, dataset_entry, chart_type)
                    run_config["model"]["chart_model"] = mc["chart_model"]
                    run_config["model"]["pretrained"] = mc.get("pretrained", False)
                    run_config["training"]["learning_rate"] = model_lr

                    _ensure_base_images(run_config)

                    _set_seeds(seed)
                    try:
                        model = train_model(run_config).to(device)
                        # torch.compile disabled — no Triton on Windows
                    except Exception as e:
                        print(f"    TRAINING FAILED: {e}")
                        accuracy_rows.append({
                            "seed": seed, "dataset": dataset_name,
                            "model": model_name, "chart_type": chart_type,
                            "augmentation": "clean", "accuracy": -1,
                        })
                        continue

                    model.eval()
                    loaders = create_dataloaders(run_config)
                    test_loader = loaders["test"]["chart"]

                    clean_acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
                    print(f"      Clean accuracy: {clean_acc:.4f}")
                    accuracy_rows.append({
                        "seed": seed, "dataset": dataset_name,
                        "model": model_name, "chart_type": chart_type,
                        "augmentation": "clean", "accuracy": clean_acc,
                    })
                    # Log clean accuracy to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/{model_name}/{chart_type}/s{seed}",
                        config={"model": model_name, "chart_type": chart_type,
                                "seed": seed, "dataset": dataset_name},
                        accuracy=clean_acc,
                    )

                    for aug_type, aug_params, aug_lbl in aug_combos:
                        def augment_fn(img, _at=aug_type, _ap=aug_params):
                            return apply_augmentation(img, _at, _ap)
                        aug_acc = _evaluate_accuracy(model, test_loader, device, augment_fn=augment_fn)
                        d = aug_acc - clean_acc
                        print(f"      {aug_lbl}: acc={aug_acc:.4f}  d={d:+.4f}")
                        accuracy_rows.append({
                            "seed": seed, "dataset": dataset_name,
                            "model": model_name, "chart_type": chart_type,
                            "augmentation": aug_lbl, "accuracy": aug_acc,
                        })

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

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — Mean Clean Accuracy by Dataset x Model")
    print(f"{'=' * 70}")
    from collections import defaultdict
    acc_map = defaultdict(list)
    for row in accuracy_rows:
        if row["augmentation"] == "clean" and row["accuracy"] >= 0:
            key = (row["dataset"], row["model"])
            acc_map[key].append(row["accuracy"])

    all_models = sorted(set(m for (_, m) in acc_map.keys()))
    header = f"{'Dataset':<22s}" + "".join(f"{m:>25s}" for m in all_models)
    print(header)
    print("-" * len(header))

    all_ds = sorted(set(ds for (ds, _) in acc_map.keys()))
    for ds in all_ds:
        row_str = f"{ds:<22s}"
        for m in all_models:
            accs = acc_map.get((ds, m), [])
            if accs:
                row_str += f"{np.mean(accs):>22.4f}+/-{np.std(accs):.3f}"
            else:
                row_str += f"{'N/A':>25s}"
        print(row_str)


def main():
    parser = argparse.ArgumentParser(description="Experiment 5I: Backbone Comparison")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
