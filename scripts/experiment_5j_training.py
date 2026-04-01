#!/usr/bin/env python
"""
Experiment 5J: Training Strategy Optimization
===============================================
Tests modern training techniques: mixup, label smoothing, cosine
annealing, class weights, and their combinations.

Uses the enhanced training loop from training_utils.py.

Usage
-----
    python scripts/experiment_5j_training.py \
        --config vtbench/config/experiment_5j_training.yaml
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

from vtbench.train.factory import get_chart_model
from vtbench.train.training_utils import train_with_enhancements
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
    output_dir = exp.get("output_dir", "results/experiment_5j_training")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])
    strategies = exp.get("training_strategies", [{"name": "baseline", "params": {}}])
    model_configs = exp.get("model_configs", [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False, "lr": 0.001}
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy_rows = []

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5j_training", config=cfg)

    for seed in seeds:
        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]

            for mc in model_configs:
                model_name = mc["name"]

                for strategy in strategies:
                    strat_name = strategy["name"]
                    strat_params = strategy.get("params", {})

                    for chart_type in chart_types:
                        print(f"\n{'=' * 60}")
                        print(f"Dataset: {dataset_name} (seed={seed})")
                        print(f"  Model: {model_name}  Strategy: {strat_name}  Chart: {chart_type}")
                        print(f"{'=' * 60}")

                        run_config = _build_run_config(cfg, dataset_entry, chart_type)
                        run_config["model"]["chart_model"] = mc["chart_model"]
                        run_config["model"]["pretrained"] = mc.get("pretrained", False)
                        run_config["training"]["learning_rate"] = mc.get("lr", 0.001)

                        # Apply strategy params
                        for k, v in strat_params.items():
                            run_config["training"][k] = v

                        _ensure_base_images(run_config)

                        _set_seeds(seed)

                        try:
                            # Create model
                            loaders = create_dataloaders(run_config)
                            train_loader = loaders['train']['chart']
                            val_loader = loaders['val']['chart']
                            test_loader = loaders['test']['chart']

                            labels = [int(label) for _, label in train_loader.dataset]
                            num_classes = len(set(labels))
                            pretrained = run_config['model'].get('pretrained', False)

                            model = get_chart_model(
                                run_config['model']['chart_model'],
                                input_channels=3,
                                num_classes=num_classes,
                                pretrained=pretrained,
                            )

                            # Train with enhancements
                            model = train_with_enhancements(
                                model, train_loader, val_loader,
                                run_config, device=device,
                            )

                        except Exception as e:
                            print(f"    TRAINING FAILED: {e}")
                            import traceback
                            traceback.print_exc()
                            accuracy_rows.append({
                                "seed": seed, "dataset": dataset_name,
                                "model": model_name, "strategy": strat_name,
                                "chart_type": chart_type, "accuracy": -1,
                            })
                            continue

                        model.eval()
                        clean_acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
                        print(f"      Clean accuracy: {clean_acc:.4f}")
                        accuracy_rows.append({
                            "seed": seed, "dataset": dataset_name,
                            "model": model_name, "strategy": strat_name,
                            "chart_type": chart_type, "accuracy": clean_acc,
                        })
                        # Log clean accuracy to W&B
                        wb.log_run_result(
                            name=f"{dataset_name}/{model_name}/{strat_name}/{chart_type}/s{seed}",
                            config={"model": model_name, "strategy": strat_name,
                                    "chart_type": chart_type, "seed": seed,
                                    "dataset": dataset_name},
                            accuracy=clean_acc,
                        )

                        del model, loaders, train_loader, val_loader, test_loader
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    fieldnames = ["seed", "dataset", "model", "strategy", "chart_type", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY — Mean Accuracy by Dataset x Model x Strategy")
    print(f"{'=' * 70}")
    from collections import defaultdict
    acc_map = defaultdict(list)
    for row in accuracy_rows:
        if row["accuracy"] >= 0:
            key = (row["dataset"], row["model"], row["strategy"])
            acc_map[key].append(row["accuracy"])

    prev_ds = None
    for (ds, mdl, strat), accs in sorted(acc_map.items()):
        if ds != prev_ds:
            print(f"\n  {ds}:")
            prev_ds = ds
        print(f"    {mdl:>12s} / {strat:<18s}:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 5J: Training Strategies")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
