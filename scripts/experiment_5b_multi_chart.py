"""
Experiment 5B: Multi-Chart Fusion Analysis
===========================================

Tests combinations of chart encodings (line+area, line+bar, line+area+bar, all 4)
to evaluate if fusion improves accuracy and robustness. Compares single-chart vs.
multi-chart models under clean and augmented conditions.

Uses the multi_modal_chart model type with different fusion_combos, training fresh
models and evaluating under augmentation sweep.

Usage
-----
    python scripts/experiment_5b_multi_chart.py \
        --config vtbench/config/experiment_5b.yaml
"""

import sys
import os

# Hack for imports (same pattern as other experiment scripts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import copy
import csv
import random
import gc

import numpy as np
import torch
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vtbench.train.trainer import train_model
from vtbench.data.loader import create_dataloaders
from vtbench.data.chart_generator import GlobalYRangeCalculator
from vtbench.utils.augmentations import apply_augmentation, augmentation_label
from vtbench.utils.wandb_logger import WandbLogger


# ====================================================================
# Helpers (shared patterns from experiment_4)
# ====================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dataset_entries(exp_cfg):
    """Parse dataset entries from experiment config."""
    exp = exp_cfg["experiment"]
    dataset_root = exp.get("dataset_root", ".")
    dataset_ext = exp.get("dataset_ext", "tsv")
    entries = []
    for item in exp.get("datasets", []):
        if isinstance(item, str):
            name = item
            train_path = os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}")
            test_path = os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}")
            entries.append({
                "name": name,
                "train_path": train_path,
                "test_path": test_path,
                "format": exp.get("dataset_format", "ucr"),
            })
        else:
            name = item["name"]
            train_path = item.get(
                "train_path",
                os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}"),
            )
            test_path = item.get(
                "test_path",
                os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}"),
            )
            entries.append({
                "name": name,
                "train_path": train_path,
                "test_path": test_path,
                "format": item.get("format", exp.get("dataset_format", "ucr")),
            })
    return entries


def _build_chart_branches_from_types(chart_types, chart_defaults):
    """Build chart_branches dict from a list of chart types.

    Parameters
    ----------
    chart_types : list of str
        Chart type names (e.g., ["line", "area", "bar"])
    chart_defaults : dict
        Default chart generation settings (color_mode, label_mode, etc.)

    Returns
    -------
    dict
        chart_branches config with branch_1, branch_2, etc.
    """
    branches = {}
    for i, chart_type in enumerate(chart_types, start=1):
        branch_key = f"branch_{i}"
        branches[branch_key] = {
            "chart_type": chart_type,
            "color_mode": chart_defaults.get("color_mode", "color"),
            "label_mode": chart_defaults.get("label_mode", "with_label"),
        }
        if chart_type == "bar":
            branches[branch_key]["bar_mode"] = chart_defaults.get("bar_mode", "border")
        if chart_type == "scatter":
            branches[branch_key]["scatter_mode"] = chart_defaults.get("scatter_mode", "plain")
    return branches


def _build_run_config(exp_cfg, dataset_entry, chart_types, fusion_combo_name):
    """Build a complete run config for a specific fusion combo.

    Parameters
    ----------
    exp_cfg : dict
        Experiment config
    dataset_entry : dict
        Dataset entry with name, train_path, test_path, format
    chart_types : list of str
        Chart types for this combo (e.g., ["line", "area"])
    fusion_combo_name : str
        Name of the fusion combo (e.g., "line_area")

    Returns
    -------
    dict
        Complete model config
    """
    chart_defaults = exp_cfg.get("chart", {})

    config = {
        "dataset": {
            "name": dataset_entry["name"],
            "train_path": dataset_entry["train_path"],
            "test_path": dataset_entry["test_path"],
            "format": dataset_entry.get("format", "ucr"),
        },
        "image_generation": copy.deepcopy(exp_cfg.get("image_generation", {})),
        "model": copy.deepcopy(exp_cfg.get("model", {})),
        "training": copy.deepcopy(exp_cfg.get("training", {})),
        "chart_branches": _build_chart_branches_from_types(chart_types, chart_defaults),
    }

    # num_classes will be set dynamically after dataloaders are created

    # Determine model type based on number of chart types
    if len(chart_types) == 1:
        config["model"]["type"] = "single_modal_chart"
    else:
        config["model"]["type"] = "multi_modal_chart"
        # multi_modal_chart requires these keys
        config["model"].setdefault("numerical_branch", "none")
        config["model"].setdefault("fusion", exp_cfg.get("experiment", {}).get("fusion_mode", "concat"))

    return config


def _ensure_base_images(config):
    """Generate and cache base chart images if needed."""
    base_cfg = copy.deepcopy(config)
    base_cfg["image_generation"].setdefault("generate_images", True)
    base_cfg["image_generation"].setdefault("overwrite_existing", False)
    create_dataloaders(base_cfg)


def _set_seeds(seed):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ====================================================================
# Multi-chart augmentation evaluation
# ====================================================================

def _evaluate_accuracy_multi_chart(model, chart_loaders, device, augment_fn=None):
    """Evaluate accuracy on multi-chart test loaders, optionally applying augmentation.

    For multi-chart models, chart_loaders is a list of DataLoaders (one per chart branch).

    Parameters
    ----------
    model : nn.Module
        Multi-chart model expecting a list of tensors
    chart_loaders : list of DataLoader
        One DataLoader per chart branch
    device : str
        torch device
    augment_fn : callable or None
        If provided, called as augment_fn(pil_image) -> pil_image for each image.

    Returns
    -------
    float
        Accuracy on test set
    """
    model.eval()
    correct = 0
    total = 0

    if augment_fn is not None:
        # Manual iteration: load images, apply augmentation, stack
        datasets = [loader.dataset for loader in chart_loaders]
        n_samples = len(datasets[0])

        with torch.no_grad():
            for idx in range(n_samples):
                # Load and augment images from all branches
                tensors = []
                label = None
                for dataset in datasets:
                    img_path = os.path.join(
                        dataset.chart_dir,
                        dataset._get_image_filename(idx),
                    )
                    pil_img = Image.open(img_path).convert("RGB")
                    pil_img = augment_fn(pil_img)
                    tensor = dataset.transform(pil_img).unsqueeze(0).to(device)
                    tensors.append(tensor)
                    if label is None:
                        label = dataset.labels[idx]

                # Forward pass with list of tensors
                logits = model(tensors)
                pred = int(torch.argmax(logits, dim=1).item())
                correct += int(pred == label)
                total += 1
    else:
        # Standard DataLoader iteration
        chart_iters = [iter(loader) for loader in chart_loaders]
        epoch_length = len(chart_loaders[0])

        with torch.no_grad():
            for batch_idx in range(epoch_length):
                try:
                    # Get batches from all chart loaders
                    chart_batches = [next(chart_iter) for chart_iter in chart_iters]

                    # Process chart data
                    tensors = []
                    label = None
                    for chart_batch in chart_batches:
                        imgs, lbls = chart_batch
                        imgs = imgs.to(device)
                        if label is None:
                            label = lbls.to(device)
                        tensors.append(imgs)

                    # Forward pass
                    logits = model(tensors)
                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == label).sum().item()
                    total += label.size(0)
                except StopIteration:
                    break

    return correct / total if total > 0 else 0.0


def _evaluate_accuracy_single_chart(model, test_loader, device, augment_fn=None):
    """Evaluate accuracy on single-chart test loader, optionally applying augmentation.

    Borrowed from experiment_4 for single-chart baselines.
    """
    model.eval()
    correct = 0
    total = 0

    if augment_fn is not None:
        # Manual iteration
        dataset = test_loader.dataset
        transform = dataset.transform
        with torch.no_grad():
            for idx in range(len(dataset)):
                img_path = os.path.join(
                    dataset.chart_dir,
                    dataset._get_image_filename(idx),
                )
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = augment_fn(pil_img)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                label = dataset.labels[idx]
                logits = model(tensor)
                pred = int(torch.argmax(logits, dim=1).item())
                correct += int(pred == label)
                total += 1
    else:
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ====================================================================
# Main experiment loop
# ====================================================================

def run_experiment(cfg):
    """Run multi-chart fusion experiment."""
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5b_multi_chart_fusion")
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="5b_multi_chart", config=cfg)

    datasets = _dataset_entries(cfg)
    seed = exp.get("seeds", [42])[0]
    fusion_mode = exp.get("fusion_mode", "concat")

    # Parse fusion combos from config
    fusion_combos = exp.get("fusion_combos", [])
    if not fusion_combos:
        # Default combos if not specified
        fusion_combos = [
            {"name": "line_only", "types": ["line"]},
            {"name": "area_only", "types": ["area"]},
            {"name": "line_area", "types": ["line", "area"]},
            {"name": "line_bar", "types": ["line", "bar"]},
            {"name": "line_area_bar", "types": ["line", "area", "bar"]},
            {"name": "all_four", "types": ["line", "area", "bar", "scatter"]},
        ]

    aug_specs = cfg.get("augmentations", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Flatten augmentation specs into (type, params, label) triples
    aug_combos = []
    for spec in aug_specs:
        aug_type = spec["type"]
        for params in spec.get("levels", [{}]):
            lbl = augmentation_label(aug_type, params)
            aug_combos.append((aug_type, dict(params), lbl))

    # Result collectors
    accuracy_rows = []

    _set_seeds(seed)

    for dataset_entry in datasets:
        dataset_name = dataset_entry["name"]
        print(f"\n{'=' * 70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 70}")

        for fusion_combo in fusion_combos:
            combo_name = fusion_combo["name"]
            chart_types = fusion_combo["types"]

            print(f"\n  Fusion Combo: {combo_name} ({', '.join(chart_types)})")

            # Build config for this combo
            run_config = _build_run_config(cfg, dataset_entry, chart_types, combo_name)

            # Ensure base images exist for all chart types
            _ensure_base_images(run_config)

            # Auto-detect num_classes from data
            tmp_loaders = create_dataloaders(run_config)
            tmp_chart = tmp_loaders['train']['chart']
            if isinstance(tmp_chart, list):
                tmp_chart = tmp_chart[0]
            labels_list = [label for _, label in tmp_chart.dataset]
            run_config['model']['num_classes'] = len(set(labels_list))

            # Train model
            _set_seeds(seed)
            model = train_model(run_config).to(device)
            model.eval()

            # Get dataloaders
            loaders = create_dataloaders(run_config)

            # For multi-chart: test_loader is a list of loaders
            # For single-chart: test_loader is a single loader
            is_multi_chart = run_config["model"]["type"] == "multi_modal_chart"

            if is_multi_chart:
                test_loaders = loaders["test"]["chart"]
                evaluate_fn = _evaluate_accuracy_multi_chart
                test_arg = test_loaders
            else:
                test_loader = loaders["test"]["chart"]
                evaluate_fn = _evaluate_accuracy_single_chart
                test_arg = test_loader

            # Evaluate clean accuracy
            clean_acc = evaluate_fn(model, test_arg, device, augment_fn=None)
            print(f"    Clean accuracy: {clean_acc:.4f}")

            accuracy_rows.append({
                "dataset": dataset_name,
                "fusion_combo": combo_name,
                "fusion_mode": fusion_mode,
                "model_type": run_config["model"]["type"],
                "num_charts": len(chart_types),
                "chart_types": ",".join(chart_types),
                "augmentation": "clean",
                "accuracy": f"{clean_acc:.4f}",
                "delta_accuracy": "0.0000",
            })

            wb.log_run_result(
                name=f"{dataset_name}/{combo_name}",
                config={"dataset": dataset_name, "fusion_combo": combo_name, "chart_types": ",".join(chart_types)},
                accuracy=clean_acc,
            )

            # Augmentation sweep
            for aug_type, aug_params, aug_lbl in aug_combos:
                def _augment(img, _at=aug_type, _ap=aug_params):
                    return apply_augmentation(img, _at, _ap)

                aug_acc = evaluate_fn(model, test_arg, device, augment_fn=_augment)
                delta = aug_acc - clean_acc
                print(f"    {aug_lbl:>20s}: acc={aug_acc:.4f}  d={delta:+.4f}")

                accuracy_rows.append({
                    "dataset": dataset_name,
                    "fusion_combo": combo_name,
                    "fusion_mode": fusion_mode,
                    "model_type": run_config["model"]["type"],
                    "num_charts": len(chart_types),
                    "chart_types": ",".join(chart_types),
                    "augmentation": aug_lbl,
                    "accuracy": f"{aug_acc:.4f}",
                    "delta_accuracy": f"{delta:.4f}",
                })

            # Write incremental CSV
            csv_path = os.path.join(output_dir, "accuracy_fusion.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset", "fusion_combo", "fusion_mode", "model_type",
                        "num_charts", "chart_types", "augmentation",
                        "accuracy", "delta_accuracy",
                    ],
                )
                writer.writeheader()
                writer.writerows(accuracy_rows)

            # Free model
            del model, loaders
            if is_multi_chart:
                del test_loaders
            else:
                del test_loader
            gc.collect()

    # ================================================================
    # Write summary markdown
    # ================================================================

    _write_summary_md(output_dir, accuracy_rows)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - accuracy_fusion.csv")
    print(f"  - summary.md")


def _write_summary_md(output_dir, accuracy_rows):
    """Generate a markdown summary table."""
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 5B: Multi-Chart Fusion Analysis — Summary\n\n")

        f.write("## Accuracy by Fusion Combo\n\n")
        f.write("| Dataset | Fusion Combo | Model Type | Num Charts | Chart Types | Augmentation | Accuracy | dAccuracy |\n")
        f.write("|---------|-------------|-----------|-----------|------------|-------------|----------|------------|\n")
        for row in accuracy_rows:
            f.write(
                f"| {row['dataset']} | {row['fusion_combo']} | {row['model_type']} "
                f"| {row['num_charts']} | {row['chart_types']} | {row['augmentation']} "
                f"| {row['accuracy']} | {row['delta_accuracy']} |\n"
            )

        f.write("\n## Key Insights\n\n")
        f.write("- Compare single-chart (baseline) vs. multi-chart (fusion) accuracy\n")
        f.write("- Measure robustness: which fusion combos degrade least under augmentation\n")
        f.write("- Analyze tradeoffs: more charts -> more parameters but potentially better generalization\n")


# ====================================================================
# CLI entry point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5B: Multi-Chart Fusion Analysis"
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
