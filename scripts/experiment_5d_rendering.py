#!/usr/bin/env python
"""
Experiment 5D: Chart Rendering Optimization
=============================================
Tests the impact of different chart rendering parameters on model
accuracy and robustness. Evaluates rendering variants such as:
  - Baseline (default rendering)
  - No labels
  - Thick lines
  - Large dots
  - High resolution (DPI)
  - Combined modifications

Each variant generates charts with modified matplotlib parameters,
trains and evaluates models independently.

Usage
-----
    python scripts/experiment_5d_rendering.py \
        --config vtbench/config/experiment_5d.yaml
"""

import argparse
import copy
import csv
import gc
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import yaml

# Setup path to import experiment_4 helpers
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from vtbench.train.trainer import train_model
from vtbench.data.loader import create_dataloaders
from vtbench.utils.augmentations import apply_augmentation, augmentation_label
from vtbench.utils.wandb_logger import WandbLogger
from vtbench.utils.heartbeat import Heartbeat
from vtbench.utils.experiment_helpers import (
    _dataset_entries,
    _build_run_config,
    _ensure_base_images,
    _evaluate_accuracy,
    _set_seeds,
)


# ====================================================================
# Rendering variant definitions
# ====================================================================

RENDER_VARIANTS = [
    {"name": "baseline", "params": {}},
    {"name": "no_labels", "params": {"label_mode": "without_label"}},
    {"name": "thick_lines", "params": {"linewidth": 3.0}},
    {"name": "large_dots", "params": {"scatter_size": 30}},
    {"name": "high_res", "params": {"dpi": 150}},
    {
        "name": "combined",
        "params": {
            "linewidth": 3.0,
            "label_mode": "without_label",
            "scatter_size": 30
        }
    },
]


# ====================================================================
# Rendering variant image generation
# ====================================================================

def _generate_variant_images(config, variant_name, variant_params):
    """
    Generate images for a rendering variant.

    For each variant, modify the dataset name in config to create separate
    image directories, ensuring variants don't overwrite each other's images.
    Also update chart config with rendering parameters.

    Parameters
    ----------
    config : dict
        Base run config
    variant_name : str
        Name of the variant (e.g., "thick_lines")
    variant_params : dict
        Rendering parameters for the variant
    """
    variant_cfg = copy.deepcopy(config)

    # Create separate dataset name for this variant to avoid cache conflicts
    original_dataset_name = variant_cfg["dataset"]["name"]
    variant_dataset_name = f"{original_dataset_name}__{variant_name}"

    variant_cfg["dataset"]["name"] = variant_dataset_name

    # Update chart config with rendering parameters
    for branch_key in list(variant_cfg["chart_branches"].keys()):
        chart_branch = variant_cfg["chart_branches"][branch_key]
        for key, value in variant_params.items():
            if key in ["label_mode", "bar_mode", "scatter_mode"]:
                chart_branch[key] = value

    # Pass rendering overrides through image_generation config
    # (picked up by build_chart_datasets -> TimeSeriesImageDataset)
    if "linewidth" in variant_params:
        variant_cfg["image_generation"]["render_linewidth"] = variant_params["linewidth"]
    if "dpi" in variant_params:
        variant_cfg["image_generation"]["render_dpi"] = variant_params["dpi"]
    if "scatter_size" in variant_params:
        variant_cfg["image_generation"]["render_scatter_s"] = variant_params["scatter_size"]

    # Generate images
    variant_cfg["image_generation"].setdefault("generate_images", True)
    variant_cfg["image_generation"].setdefault("overwrite_existing", True)

    create_dataloaders(variant_cfg)

    return variant_dataset_name


# ====================================================================
# Main experiment loop
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5d_rendering")
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="5d_rendering", config=cfg)
    hb = Heartbeat("5d")

    chart_types = exp.get("chart_types", ["line", "area", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])

    # Render variants from config or use defaults
    render_variants = exp.get("render_variants", RENDER_VARIANTS)

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

    total_runs = len(seeds) * len(datasets) * len(chart_types) * len(render_variants)
    run_idx = 0

    for seed in seeds:
        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name} (seed={seed})")
            print(f"{'=' * 60}")

            for chart_type in chart_types:
                print(f"\n  Encoding: {chart_type}")

                for variant in render_variants:
                    run_idx += 1
                    variant_name = variant["name"]
                    variant_params = variant["params"]

                    print(f"    Variant: {variant_name}")

                    hb.pulse(
                        dataset=dataset_name,
                        encoding=chart_type,
                        model="chart_model",
                        run=f"{run_idx}/{total_runs}",
                        variant=variant_name,
                    )

                    run_config = _build_run_config(cfg, dataset_entry, chart_type)

                    # Generate or fetch images for this variant
                    variant_dataset_name = _generate_variant_images(
                        run_config, variant_name, variant_params
                    )

                    # Update dataset name in config for this variant
                    run_config["dataset"]["name"] = variant_dataset_name

                    _set_seeds(seed)
                    model = train_model(run_config).to(device)
                    model.eval()

                    loaders = create_dataloaders(run_config)
                    test_loader = loaders["test"]["chart"]

                    # Compute clean baseline accuracy
                    clean_acc = _evaluate_accuracy(
                        model, test_loader, device, augment_fn=None
                    )
                    print(f"      Clean accuracy: {clean_acc:.4f}")

                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "variant": variant_name,
                        "augmentation": "clean",
                        "accuracy": f"{clean_acc:.4f}",
                        "delta_accuracy": "0.0000",
                    })

                    wb.log_run_result(
                        name=f"{dataset_name}/{chart_type}/{variant_name}/s{seed}",
                        config={"dataset": dataset_name, "encoding": chart_type, "variant": variant_name, "seed": seed},
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
                        print(
                            f"      {aug_lbl:>20s}: acc={aug_acc:.4f}  "
                            f"d={delta:+.4f}"
                        )

                        accuracy_rows.append({
                            "seed": seed,
                            "dataset": dataset_name,
                            "encoding": chart_type,
                            "variant": variant_name,
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

    acc_csv = os.path.join(output_dir, "accuracy_rendering_variants.csv")
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "dataset", "encoding", "variant",
                "augmentation", "accuracy", "delta_accuracy"
            ],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nWrote results to {acc_csv}")

    # Write summary markdown
    _write_summary_md(output_dir, accuracy_rows)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - accuracy_rendering_variants.csv")
    print(f"  - summary.md")

    hb.close()


def _write_summary_md(output_dir, accuracy_rows):
    """Generate a markdown summary for quick inspection."""
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 5D: Chart Rendering Optimization — Summary\n\n")

        f.write("## Accuracy by Rendering Variant\n\n")
        f.write(
            "| Seed | Dataset | Encoding | Variant | "
            "Augmentation | Accuracy | dAccuracy |\n"
        )
        f.write(
            "|------|---------|----------|---------|"
            "|--------------|----------|----------|\n"
        )
        for row in accuracy_rows:
            f.write(
                f"| {row['seed']} | {row['dataset']} | {row['encoding']} | "
                f"{row['variant']} | {row['augmentation']} | {row['accuracy']} | "
                f"{row['delta_accuracy']} |\n"
            )


# ====================================================================
# Config loading and CLI entry point
# ====================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5D: Chart Rendering Optimization"
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
