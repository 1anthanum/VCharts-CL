"""
Experiment 5A: Training-Time Augmentation
==========================================
Evaluates the effect of random on-the-fly chart augmentations during training
on model robustness. Tests three augmentation regimes: Mild, Moderate, Heavy.

During training, images are randomly augmented on-the-fly (50% chance per batch
item) with: blur, noise, JPEG compress, crop, or stripe occlusion.
Evaluates clean accuracy and augmented accuracy post-training.

Usage
-----
    python scripts/experiment_5a_train_augmentation.py \
        --config vtbench/config/experiment_5a.yaml
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
from torch.utils.data import Dataset, DataLoader

# Setup path to import experiment_4 helpers
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from vtbench.train.trainer import train_model, train_standard_model
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


# ====================================================================
# AugmentedDatasetWrapper: applies random augmentations on-the-fly
# ====================================================================

class AugmentedDatasetWrapper(Dataset):
    """Wraps a TimeSeriesImageDataset and applies random augmentations
    during training.

    For each __getitem__ call:
    1. Load the image from the original dataset's chart_dir
    2. Apply a random augmentation (with 50% probability)
    3. Apply the original transform

    Parameters
    ----------
    dataset : TimeSeriesImageDataset
        The underlying chart image dataset.
    regime : str or None
        One of: "mild", "moderate", "heavy", or None.
        Controls augmentation intensity:
        - mild: blur sigma~U[0,0.5], noise sigma~U[0,10], jpeg q~U[50,100]
        - moderate: blur sigma~U[0,1.0], noise sigma~U[0,25], jpeg q~U[20,100]
        - heavy: blur sigma~U[0,2.0], noise sigma~U[0,40], crop_frac~U[0,0.10],
                 plus 0-5 stripes
        When None, no augmentation is applied (baseline).
    """

    def __init__(self, dataset, regime=None):
        self.dataset = dataset
        self.regime = regime
        self.chart_dir = dataset.chart_dir
        self.transform = dataset.transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load raw image
        img_path = os.path.join(
            self.chart_dir,
            self.dataset._get_image_filename(idx),
        )
        pil_img = Image.open(img_path).convert("RGB")

        # Apply random augmentation with 50% probability
        if self.regime is not None and random.random() < 0.5:
            pil_img = self._apply_random_augmentation(pil_img)

        # Apply the original transform
        if self.transform:
            tensor = self.transform(pil_img)
        else:
            tensor = pil_img

        label = torch.tensor(self.dataset.labels[idx], dtype=torch.long)
        return tensor, label

    def _apply_random_augmentation(self, img):
        """Apply a random augmentation based on regime."""
        if self.regime is None:
            return img

        # Select random augmentation type
        aug_type = np.random.choice([
            "gaussian_blur", "gaussian_noise", "jpeg_compress",
            "crop_horizontal", "stripe_mask"
        ])

        if self.regime == "mild":
            params = self._get_mild_params(aug_type)
        elif self.regime == "moderate":
            params = self._get_moderate_params(aug_type)
        elif self.regime == "heavy":
            params = self._get_heavy_params(aug_type)
        else:
            raise ValueError(f"Unknown regime: {self.regime}")

        return apply_augmentation(img, aug_type, params)

    @staticmethod
    def _get_mild_params(aug_type):
        """Mild augmentation parameters."""
        if aug_type == "gaussian_blur":
            sigma = np.random.uniform(0, 0.5)
            return {"sigma": sigma}
        elif aug_type == "gaussian_noise":
            sigma = np.random.uniform(0, 10)
            return {"sigma": sigma, "seed": np.random.randint(0, 2**16)}
        elif aug_type == "jpeg_compress":
            quality = np.random.uniform(50, 100)
            return {"quality": int(quality)}
        elif aug_type == "crop_horizontal":
            crop_frac = np.random.uniform(0, 0.05)
            return {"crop_fraction": crop_frac}
        elif aug_type == "stripe_mask":
            num_stripes = np.random.randint(0, 3)
            return {"num_stripes": num_stripes, "stripe_width": 2}
        return {}

    @staticmethod
    def _get_moderate_params(aug_type):
        """Moderate augmentation parameters."""
        if aug_type == "gaussian_blur":
            sigma = np.random.uniform(0, 1.0)
            return {"sigma": sigma}
        elif aug_type == "gaussian_noise":
            sigma = np.random.uniform(0, 25)
            return {"sigma": sigma, "seed": np.random.randint(0, 2**16)}
        elif aug_type == "jpeg_compress":
            quality = np.random.uniform(20, 100)
            return {"quality": int(quality)}
        elif aug_type == "crop_horizontal":
            crop_frac = np.random.uniform(0, 0.08)
            return {"crop_fraction": crop_frac}
        elif aug_type == "stripe_mask":
            num_stripes = np.random.randint(0, 4)
            return {"num_stripes": num_stripes, "stripe_width": 3}
        return {}

    @staticmethod
    def _get_heavy_params(aug_type):
        """Heavy augmentation parameters."""
        if aug_type == "gaussian_blur":
            sigma = np.random.uniform(0, 2.0)
            return {"sigma": sigma}
        elif aug_type == "gaussian_noise":
            sigma = np.random.uniform(0, 40)
            return {"sigma": sigma, "seed": np.random.randint(0, 2**16)}
        elif aug_type == "jpeg_compress":
            quality = np.random.uniform(10, 100)
            return {"quality": int(quality)}
        elif aug_type == "crop_horizontal":
            crop_frac = np.random.uniform(0, 0.10)
            return {"crop_fraction": crop_frac}
        elif aug_type == "stripe_mask":
            num_stripes = np.random.randint(0, 6)
            return {"num_stripes": num_stripes, "stripe_width": 4}
        return {}


# ====================================================================
# Helpers for training with augmented datasets
# ====================================================================

def _get_regime_name(regime):
    """Return a display name for a regime."""
    if regime is None:
        return "baseline"
    return regime.lower()


# ====================================================================
# Main experiment loop
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5a_train_augmentation")
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="5a_train_augmentation", config=cfg)

    chart_types = exp.get("chart_types", ["line", "area"])
    datasets = _dataset_entries(cfg)
    regimes = exp.get("regimes", [None, "mild", "moderate", "heavy"])
    seeds = exp.get("seeds", [42])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Result collectors
    accuracy_rows = []
    augmentation_sweep_rows = []

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

                for regime in regimes:
                    regime_name = _get_regime_name(regime)
                    print(f"    Regime: {regime_name}")

                    _set_seeds(seed)

                    # Create dataloaders
                    loaders = create_dataloaders(run_config)
                    train_loader = loaders["train"]["chart"]
                    val_loader = loaders["val"]["chart"]
                    test_loader = loaders["test"]["chart"]

                    # Wrap train and val loaders with augmentation if regime is not None
                    if regime is not None:
                        train_wrapped = AugmentedDatasetWrapper(
                            train_loader.dataset,
                            regime=regime
                        )
                        train_loader = DataLoader(
                            train_wrapped,
                            batch_size=train_loader.batch_size,
                            shuffle=True,
                            drop_last=True,
                        )

                        val_wrapped = AugmentedDatasetWrapper(
                            val_loader.dataset,
                            regime=regime
                        )
                        val_loader = DataLoader(
                            val_wrapped,
                            batch_size=val_loader.batch_size,
                            shuffle=False,
                        )

                    # Train model using custom loaders (with augmentation wrapper)
                    _set_seeds(seed)
                    # Unwrap loader if list
                    if isinstance(train_loader, list):
                        train_loader = train_loader[0]
                    if isinstance(val_loader, list):
                        val_loader = val_loader[0]
                    if isinstance(test_loader, list):
                        test_loader = test_loader[0]
                    # Create model manually and train with our augmented loaders
                    labels_list = [label for _, label in train_loader.dataset]
                    num_classes = len(set(labels_list))
                    pretrained = run_config['model'].get('pretrained', False)
                    chart_model_name = run_config['model']['chart_model']
                    model = get_chart_model(chart_model_name, input_channels=3,
                                            num_classes=num_classes, pretrained=pretrained)
                    model = train_standard_model(model, train_loader, val_loader,
                                                 test_loader, run_config).to(device)
                    model.eval()

                    # Evaluate clean accuracy
                    clean_acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
                    print(f"      Clean accuracy: {clean_acc:.4f}")

                    accuracy_rows.append({
                        "seed": seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "regime": regime_name,
                        "augmentation": "clean",
                        "accuracy": f"{clean_acc:.4f}",
                        "delta_accuracy": "0.0000",
                    })

                    wb.log_run_result(
                        name=f"{dataset_name}/{chart_type}/{regime_name}/s{seed}",
                        config={"dataset": dataset_name, "encoding": chart_type, "regime": regime_name, "seed": seed},
                        accuracy=clean_acc,
                    )

                    # Augmentation sweep evaluation
                    aug_specs = cfg.get("augmentations", [])
                    aug_combos = []
                    for spec in aug_specs:
                        aug_type = spec["type"]
                        for params in spec.get("levels", [{}]):
                            lbl = augmentation_label(aug_type, params)
                            aug_combos.append((aug_type, dict(params), lbl))

                    for aug_type, aug_params, aug_lbl in aug_combos:
                        def _augment(img, _at=aug_type, _ap=aug_params):
                            return apply_augmentation(img, _at, _ap)

                        aug_acc = _evaluate_accuracy(
                            model, test_loader, device, augment_fn=_augment
                        )
                        delta = aug_acc - clean_acc
                        print(f"      {aug_lbl:>20s}: acc={aug_acc:.4f}  d={delta:+.4f}")

                        augmentation_sweep_rows.append({
                            "seed": seed,
                            "dataset": dataset_name,
                            "encoding": chart_type,
                            "regime": regime_name,
                            "augmentation": aug_lbl,
                            "accuracy": f"{aug_acc:.4f}",
                            "delta_accuracy": f"{delta:.4f}",
                        })

                    # Memory cleanup
                    del model, loaders, train_loader, val_loader, test_loader
                    gc.collect()

    # ================================================================
    # Write summary CSVs
    # ================================================================

    acc_csv = os.path.join(output_dir, "clean_accuracy_by_regime.csv")
    with open(acc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "dataset", "encoding", "regime",
                "augmentation", "accuracy", "delta_accuracy"
            ],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nWrote clean accuracy results to {acc_csv}")

    aug_csv = os.path.join(output_dir, "augmentation_robustness_by_regime.csv")
    with open(aug_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed", "dataset", "encoding", "regime",
                "augmentation", "accuracy", "delta_accuracy"
            ],
        )
        writer.writeheader()
        writer.writerows(augmentation_sweep_rows)
    print(f"Wrote augmentation robustness results to {aug_csv}")

    # Write summary markdown
    _write_summary_md(output_dir, accuracy_rows, augmentation_sweep_rows)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - clean_accuracy_by_regime.csv")
    print(f"  - augmentation_robustness_by_regime.csv")
    print(f"  - summary.md")


def _write_summary_md(output_dir, accuracy_rows, augmentation_rows):
    """Generate a markdown summary for quick inspection."""
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 5A: Training-Time Augmentation — Summary\n\n")

        f.write("## Clean Accuracy by Regime\n\n")
        f.write("| Seed | Dataset | Encoding | Regime | Accuracy |\n")
        f.write("|------|---------|----------|--------|----------|\n")
        for row in accuracy_rows:
            if row["augmentation"] == "clean":
                f.write(
                    f"| {row['seed']} | {row['dataset']} | {row['encoding']} | "
                    f"{row['regime']} | {row['accuracy']} |\n"
                )

        f.write("\n## Augmentation Robustness by Regime\n\n")
        f.write("| Seed | Dataset | Encoding | Regime | Augmentation | Accuracy | dAccuracy |\n")
        f.write("|------|---------|----------|--------|--------------|----------|----------|\n")
        for row in augmentation_rows:
            f.write(
                f"| {row['seed']} | {row['dataset']} | {row['encoding']} | "
                f"{row['regime']} | {row['augmentation']} | {row['accuracy']} | {row['delta_accuracy']} |\n"
            )


# ====================================================================
# CLI entry point
# ====================================================================

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 5A: Training-Time Augmentation"
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
