#!/usr/bin/env python
"""
Experiment 5E: Two-Branch Numerical Fusion
===========================================
Evaluates multimodal fusion of chart images with raw numerical features
using two-branch architectures. Tests different numerical encoders
(FCN, Transformer, OSCNN) combined with chart encoders.

For each (dataset, chart_type, numerical_model) combination:
  1. Train a two-branch model with the specified components
  2. Evaluate clean accuracy
  3. Evaluate robustness under test-time image augmentations
  4. Record results

Note: Image augmentations are applied only to the chart branch;
numerical data remains unmodified.

Usage
-----
    python scripts/experiment_5e_two_branch.py \
        --config vtbench/config/experiment_5e.yaml
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
    _set_seeds,
)


# ====================================================================
# Two-branch augmented evaluation
# ====================================================================

def _evaluate_two_branch_augmented(
    model, chart_loader, numerical_data, device, augment_fn=None
):
    """
    Evaluate a two-branch model with optional augmentation applied to
    chart images only.

    Parameters
    ----------
    model : nn.Module
        Two-branch model that expects input as (chart_tensor, num_tensor)
    chart_loader : DataLoader
        Chart image dataloader for test set
    numerical_data : torch.Tensor
        Numerical features tensor for test set (shape: [N, feature_dim])
    device : str
        Device to use ("cuda" or "cpu")
    augment_fn : callable or None
        If provided, called as augment_fn(pil_image) -> pil_image for
        each chart sample.

    Returns
    -------
    float
        Accuracy on the test set
    """
    model.eval()
    correct = 0
    total = 0

    dataset = chart_loader.dataset
    transform = dataset.transform

    if augment_fn is not None:
        # Manual iteration with augmentation applied only to chart images
        # numerical_data is a DataLoader; access its underlying dataset
        num_dataset = numerical_data.dataset if hasattr(numerical_data, 'dataset') else numerical_data
        with torch.no_grad():
            for idx in range(len(dataset)):
                # Load and augment chart image
                img_path = os.path.join(
                    dataset.chart_dir,
                    dataset._get_image_filename(idx),
                )
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = augment_fn(pil_img)
                chart_tensor = transform(pil_img).unsqueeze(0).to(device)

                # Numerical features (dataset returns (tensor, label) tuples)
                num_item = num_dataset[idx]
                num_tensor = (num_item[0] if isinstance(num_item, (tuple, list)) else num_item).unsqueeze(0).to(device)

                # Forward pass with tuple input
                logits = model((chart_tensor, num_tensor))
                pred = int(torch.argmax(logits, dim=1).item())
                label = dataset.labels[idx]
                correct += int(pred == label)
                total += 1
    else:
        # Standard loader-based iteration (no augmentation)
        with torch.no_grad():
            num_iter = iter(numerical_data)
            for chart_batch, labels in chart_loader:
                chart_batch = chart_batch.to(device)
                try:
                    num_batch, _ = next(num_iter)
                except StopIteration:
                    break
                num_batch = num_batch.to(device)

                logits = model((chart_batch, num_batch))
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ====================================================================
# Main experiment loop
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5e_two_branch")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area"])
    datasets = _dataset_entries(cfg)
    numerical_models = exp.get("numerical_models", ["fcn", "oscnn"])
    seeds = exp.get("seeds", [42])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5e_two_branch", config=cfg)

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

                # Load dataloaders to get numerical features
                loaders = create_dataloaders(run_config)
                test_chart_loader = loaders["test"]["chart"]
                # Unwrap if list (single_modal_chart can return list with one element)
                if isinstance(test_chart_loader, list):
                    test_chart_loader = test_chart_loader[0]
                test_numerical_data = loaders["test"].get("numerical")

                # Auto-detect num_classes from training data
                train_chart = loaders["train"]["chart"]
                if isinstance(train_chart, list):
                    train_chart = train_chart[0]
                _labels = [label for _, label in train_chart.dataset]
                detected_num_classes = len(set(_labels))

                for numerical_model in numerical_models:
                    print(f"    Numerical: {numerical_model}")

                    # Update config for two-branch model
                    branch_config = copy.deepcopy(run_config)
                    branch_config["model"]["type"] = "two_branch"
                    branch_config["model"]["numerical_branch"] = numerical_model
                    branch_config["model"]["num_classes"] = detected_num_classes
                    branch_config["model"].setdefault("fusion", cfg.get("experiment", {}).get("fusion_mode", "concat"))

                    # Train model
                    _set_seeds(seed)
                    model = train_model(branch_config).to(device)
                    model.eval()

                    # Evaluate clean accuracy
                    if test_numerical_data is not None:
                        clean_acc = _evaluate_two_branch_augmented(
                            model, test_chart_loader, test_numerical_data,
                            device, augment_fn=None
                        )
                    else:
                        # Fallback if numerical data not available
                        with torch.no_grad():
                            correct = 0
                            total = 0
                            for images, labels in test_chart_loader:
                                images = images.to(device)
                                logits = model(images)
                                preds = torch.argmax(logits, dim=1)
                                correct += (preds == labels.to(device)).sum().item()
                                total += labels.size(0)
                        clean_acc = correct / total if total > 0 else 0.0

                    print(f"      Clean accuracy: {clean_acc:.4f}")

                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "numerical_model": numerical_model,
                        "augmentation": "clean",
                        "accuracy": f"{clean_acc:.4f}",
                        "delta_accuracy": "0.0000",
                    })

                    # Log to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}_{chart_type}_{numerical_model}_seed{seed}",
                        config={
                            "dataset": dataset_name,
                            "encoding": chart_type,
                            "numerical_model": numerical_model,
                            "seed": seed,
                        },
                        accuracy=clean_acc,
                    )

                    # Augmentation sweep (chart images only)
                    if test_numerical_data is not None:
                        for aug_type, aug_params, aug_lbl in aug_combos:
                            def _augment(img, _at=aug_type, _ap=aug_params):
                                return apply_augmentation(img, _at, _ap)

                            aug_acc = _evaluate_two_branch_augmented(
                                model, test_chart_loader, test_numerical_data,
                                device, augment_fn=_augment
                            )
                            delta = aug_acc - clean_acc
                            print(
                                f"      {aug_lbl:>20s}: acc={aug_acc:.4f}  "
                                f"d={delta:+.4f}"
                            )

                            accuracy_rows.append({
                                "seed": seed,
                                "dataset": dataset_name,
                                "encoding": chart_type,
                                "numerical_model": numerical_model,
                                "augmentation": aug_lbl,
                                "accuracy": f"{aug_acc:.4f}",
                                "delta_accuracy": f"{delta:.4f}",
                            })

                    # Memory cleanup
                    del model
                    gc.collect()

                # Clean up loaders after all numerical models
                del loaders, test_chart_loader, test_numerical_data
                gc.collect()

    # ================================================================
    # Write summary CSV
    # ================================================================

    acc_csv = os.path.join(output_dir, "accuracy_two_branch.csv")
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "dataset", "encoding", "numerical_model",
                "augmentation", "accuracy", "delta_accuracy"
            ],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nWrote results to {acc_csv}")

    # Write summary markdown
    _write_summary_md(output_dir, accuracy_rows)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - accuracy_two_branch.csv")
    print(f"  - summary.md")


def _write_summary_md(output_dir, accuracy_rows):
    """Generate a markdown summary for quick inspection."""
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 5E: Two-Branch Numerical Fusion — Summary\n\n")

        f.write("## Accuracy by Numerical Model\n\n")
        f.write(
            "| Seed | Dataset | Encoding | Numerical | "
            "Augmentation | Accuracy | dAccuracy |\n"
        )
        f.write(
            "|------|---------|----------|-----------|"
            "|--------------|----------|----------|\n"
        )
        for row in accuracy_rows:
            f.write(
                f"| {row['seed']} | {row['dataset']} | {row['encoding']} | "
                f"{row['numerical_model']} | {row['augmentation']} | "
                f"{row['accuracy']} | {row['delta_accuracy']} |\n"
            )


# ====================================================================
# Config loading and CLI entry point
# ====================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5E: Two-Branch Numerical Fusion"
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
