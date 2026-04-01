"""
Shared helper functions for experiment scripts.

Extracted from experiment_4_augmentation_robustness.py which was implicitly
used as a library by 12+ experiment scripts. Now all experiments can import
from this proper module instead.

Usage:
    from vtbench.utils.experiment_helpers import (
        load_config, dataset_entries, build_run_config,
        ensure_base_images, set_seeds, evaluate_accuracy,
    )
"""

import copy
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image

from vtbench.data.loader import create_dataloaders


def load_config(path):
    """Load a YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def dataset_entries(exp_cfg):
    """Parse dataset list from experiment config.

    Handles both simple string format and dict format with custom paths.

    Args:
        exp_cfg: Full experiment config dict (must contain 'experiment' key)

    Returns:
        list of dicts with 'name', 'train_path', 'test_path', 'format' keys
    """
    exp = exp_cfg["experiment"]
    dataset_root = exp.get("dataset_root", ".")
    dataset_ext = exp.get("dataset_ext", "tsv")
    entries = []

    for item in exp.get("datasets", []):
        if isinstance(item, str):
            name = item
            train_path = os.path.join(
                dataset_root, name, f"{name}_TRAIN.{dataset_ext}"
            )
            test_path = os.path.join(
                dataset_root, name, f"{name}_TEST.{dataset_ext}"
            )
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


def build_run_config(exp_cfg, dataset_entry, chart_type):
    """Build a single-run config dict from experiment config + dataset + chart type.

    This creates the config structure expected by train_model() and create_dataloaders().
    """
    chart_defaults = exp_cfg.get("chart", {})
    bar_mode = chart_defaults.get("bar_mode", "border")
    scatter_mode = chart_defaults.get("scatter_mode", "plain")

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
        "chart_branches": {
            "branch_1": {
                "chart_type": chart_type,
                "color_mode": chart_defaults.get("color_mode", "color"),
                "label_mode": chart_defaults.get("label_mode", "with_label"),
            }
        },
    }
    if chart_type == "bar":
        config["chart_branches"]["branch_1"]["bar_mode"] = bar_mode
    if chart_type == "scatter":
        config["chart_branches"]["branch_1"]["scatter_mode"] = scatter_mode
    return config


def ensure_base_images(config):
    """Generate chart images if they don't exist yet."""
    base_cfg = copy.deepcopy(config)
    base_cfg["image_generation"].setdefault("generate_images", True)
    base_cfg["image_generation"].setdefault("overwrite_existing", False)
    create_dataloaders(base_cfg)


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def evaluate_accuracy(model, test_loader, device, augment_fn=None):
    """Evaluate classification accuracy on a test loader.

    Args:
        model: nn.Module
        test_loader: DataLoader
        device: torch device string
        augment_fn: Optional callable (PIL Image -> PIL Image) for test-time augmentation

    Returns:
        float: accuracy in [0, 1]
    """
    model.eval()
    correct = 0
    total = 0

    if augment_fn is not None:
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


# Backward-compatible aliases (for scripts that use the underscore-prefixed names)
_dataset_entries = dataset_entries
_build_run_config = build_run_config
_ensure_base_images = ensure_base_images
_set_seeds = set_seeds
_evaluate_accuracy = evaluate_accuracy
