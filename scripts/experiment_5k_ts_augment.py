#!/usr/bin/env python
"""
Experiment 5K: Time-Series Level Data Augmentation
====================================================
Augments raw time-series BEFORE chart generation, creating new training
samples. Tests data expansion multipliers: 0 (original), 2x, 5x, 10x.

Pre-generates augmented chart images, then trains using a custom
DataLoader that directly reads from the pre-generated image directory.

Usage
-----
    # 1. Pre-generate images first:
    python scripts/pregenerate_5k_images.py

    # 2. Run experiment:
    python scripts/experiment_5k_ts_augment.py \
        --config vtbench/config/experiment_5k_ts_augment.yaml
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
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from vtbench.data.loader import read_ucr, create_dataloaders
from vtbench.data.chart_generator import (
    create_line_chart, create_scatter_chart, GlobalYRangeCalculator,
)
from vtbench.train.factory import get_chart_model
from vtbench.utils.ts_augmentations import augment_dataset
from vtbench.utils.wandb_logger import WandbLogger
from vtbench.utils.heartbeat import Heartbeat
from vtbench.utils.experiment_helpers import (
    _dataset_entries,
    _build_run_config,
    _ensure_base_images,
    _evaluate_accuracy,
    _set_seeds,
)


CHART_FN = {
    "line": create_line_chart,
    "scatter": create_scatter_chart,
}


# ====================================================================
# Custom Dataset for pre-generated augmented images
# ====================================================================

class AugmentedImageDataset(Dataset):
    """Reads chart images from a directory of sample_0.png, sample_1.png, ..."""

    def __init__(self, img_dir, labels, transform=None):
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, f"sample_{idx}.png")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ====================================================================
# Image generation helper (inline, for on-the-fly generation)
# ====================================================================

def _ensure_augmented_images(X_aug, y_aug, dataset_name, chart_type,
                             multiplier, seed, chart_defaults, image_root):
    """Generate chart images for augmented time-series if not already present."""
    dir_name = f"{dataset_name}_aug{multiplier}x_s{seed}_images"
    img_dir = os.path.join(image_root, dir_name, chart_type)
    os.makedirs(img_dir, exist_ok=True)

    chart_fn = CHART_FN.get(chart_type)
    if chart_fn is None:
        raise ValueError(f"Unsupported chart type for 5K: {chart_type}")

    # Check if already complete
    expected = len(X_aug)
    existing = sum(1 for f in os.listdir(img_dir) if f.endswith(".png"))
    if existing >= expected:
        print(f"    Images exist: {img_dir} ({existing} files)")
        return img_dir

    # Generate
    global_y_range = GlobalYRangeCalculator.calculate_global_y_range_univariate(list(X_aug))

    color_mode = chart_defaults.get("color_mode", "color")
    label_mode = chart_defaults.get("label_mode", "with_label")

    generated = 0
    for i in range(len(X_aug)):
        path = os.path.join(img_dir, f"sample_{i}.png")
        if os.path.exists(path):
            continue
        kwargs = {"color_mode": color_mode, "label_mode": label_mode,
                  "global_y_range": global_y_range}
        if chart_type == "scatter":
            kwargs["scatter_mode"] = chart_defaults.get("scatter_mode", "plain")
        chart_fn(X_aug[i], path, **kwargs)
        plt.close("all")
        generated += 1

    print(f"    Generated {generated} new images in {img_dir}")
    return img_dir


# ====================================================================
# Training helper (bypasses create_dataloaders)
# ====================================================================

def _train_from_images(model, train_dir, train_labels, val_dir, val_labels,
                       config, device):
    """Train model directly from image directories."""
    img_size = config.get("training", {}).get("image_size", 128)
    pretrained = config.get("model", {}).get("pretrained", False)

    base_transforms = [transforms.Resize((img_size, img_size)), transforms.ToTensor()]
    if pretrained:
        base_transforms.append(transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    tf = transforms.Compose(base_transforms)

    train_ds = AugmentedImageDataset(train_dir, train_labels, transform=tf)
    val_ds = AugmentedImageDataset(val_dir, val_labels, transform=tf)

    batch_size = config.get("training", {}).get("batch_size", 64)
    safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
    train_loader = DataLoader(train_ds, batch_size=safe_bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False)

    lr = config.get("training", {}).get("learning_rate", 0.001)
    epochs = config.get("training", {}).get("epochs", 100)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    trigger_times = 0
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        val_loss_avg = val_loss / max(len(val_loader), 1)

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.1f}%, "
              f"Val Acc: {val_acc:.1f}%, Loss: {val_loss_avg:.4f}")

        scheduler.step(val_loss_avg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model


# ====================================================================
# Main experiment
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5k_ts_augment")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])
    multipliers = exp.get("augment_multipliers", [0, 2, 5, 10])
    ts_aug_names = exp.get("ts_augmentations",
                           ["jitter", "scaling", "time_warp", "magnitude_warp"])
    chart_defaults = cfg.get("chart", {})

    _img_root = os.environ.get("CHART_IMAGE_ROOT", "chart_images")
    image_root = os.path.realpath(_img_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy_rows = []

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5k_ts_augment", config=cfg)
    hb = Heartbeat("5k")

    total_runs = len(seeds) * len(datasets) * len(multipliers) * len(chart_types)
    run_idx = 0

    for seed in seeds:
        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]

            # Load raw data (read_ucr takes a single file path)
            X_train, y_train = read_ucr(dataset_entry["train_path"])
            num_classes = len(set(y_train.tolist()))

            for multiplier in multipliers:
                for chart_type in chart_types:
                    run_idx += 1

                    # --- multiplier=0: use original pipeline ---
                    if multiplier == 0:
                        print(f"\n{'=' * 60}")
                        print(f"Dataset: {dataset_name} (seed={seed}, original, {chart_type})")
                        print(f"{'=' * 60}")

                        hb.pulse(
                            dataset=dataset_name,
                            encoding=chart_type,
                            model="chart_model",
                            run=f"{run_idx}/{total_runs}",
                            multiplier=multiplier,
                        )

                        run_config = _build_run_config(cfg, dataset_entry, chart_type)
                        _ensure_base_images(run_config)
                        _set_seeds(seed)

                        try:
                            from vtbench.train.trainer import train_model
                            model = train_model(run_config).to(device)
                            model.eval()
                            loaders = create_dataloaders(run_config)
                            test_loader = loaders["test"]["chart"]
                            clean_acc = _evaluate_accuracy(
                                model, test_loader, device, augment_fn=None)
                        except Exception as e:
                            print(f"    FAILED: {e}")
                            clean_acc = -1

                        print(f"    Clean accuracy: {clean_acc:.4f}")
                        accuracy_rows.append({
                            "seed": seed, "dataset": dataset_name,
                            "multiplier": 0, "chart_type": chart_type,
                            "train_size": len(X_train), "accuracy": clean_acc,
                        })
                        # Log clean accuracy to W&B
                        wb.log_run_result(
                            name=f"{dataset_name}/original/{chart_type}/s{seed}",
                            config={"multiplier": 0, "chart_type": chart_type,
                                    "train_size": len(X_train), "seed": seed,
                                    "dataset": dataset_name},
                            accuracy=clean_acc,
                        )
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue

                    # --- multiplier > 0: augmented pipeline ---
                    X_aug, y_aug = augment_dataset(
                        X_train, y_train,
                        num_augmented=multiplier,
                        aug_names=ts_aug_names,
                        seed=seed,
                    )

                    print(f"\n{'=' * 60}")
                    print(f"Dataset: {dataset_name} (seed={seed}, "
                          f"aug_{multiplier}x, {chart_type})")
                    print(f"  Samples: {len(X_train)} -> {len(X_aug)}")
                    print(f"{'=' * 60}")

                    hb.pulse(
                        dataset=dataset_name,
                        encoding=chart_type,
                        model="chart_model",
                        run=f"{run_idx}/{total_runs}",
                        multiplier=multiplier,
                    )

                    # Ensure augmented images exist
                    img_dir = _ensure_augmented_images(
                        X_aug, y_aug, dataset_name, chart_type,
                        multiplier, seed, chart_defaults, image_root,
                    )

                    # Split augmented data into train/val (80/20)
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                                random_state=seed)
                    train_idx, val_idx = next(sss.split(X_aug, y_aug))
                    y_aug_tensor = torch.tensor(y_aug, dtype=torch.long)

                    train_labels = y_aug_tensor[train_idx]
                    val_labels = y_aug_tensor[val_idx]

                    # Create subset image directories (symlinks or index mapping)
                    # Since AugmentedImageDataset reads by index, we need to
                    # remap indices. Simpler: create a wrapper dataset.
                    class SubsetAugDataset(Dataset):
                        def __init__(self, img_dir, indices, labels, transform):
                            self.img_dir = img_dir
                            self.indices = indices
                            self.labels = labels
                            self.transform = transform

                        def __len__(self):
                            return len(self.indices)

                        def __getitem__(self, idx):
                            real_idx = self.indices[idx]
                            path = os.path.join(self.img_dir, f"sample_{real_idx}.png")
                            try:
                                img = Image.open(path).convert("RGB")
                            except Exception:
                                img = Image.new("RGB", (128, 128), (0, 0, 0))
                            if self.transform:
                                img = self.transform(img)
                            return img, self.labels[idx]

                    # Build transform
                    img_size = cfg.get("training", {}).get("image_size", 128)
                    tf = transforms.Compose([
                        transforms.Resize((img_size, img_size)),
                        transforms.ToTensor(),
                    ])

                    # Build model
                    _set_seeds(seed)
                    pretrained = cfg["model"].get("pretrained", False)
                    model = get_chart_model(
                        cfg["model"]["chart_model"],
                        input_channels=3,
                        num_classes=num_classes,
                        pretrained=pretrained,
                    )

                    batch_size = cfg.get("training", {}).get("batch_size", 64)
                    train_ds = SubsetAugDataset(img_dir, train_idx, train_labels, tf)
                    val_ds = SubsetAugDataset(img_dir, val_idx, val_labels, tf)
                    safe_bs2 = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                    train_loader = DataLoader(train_ds, batch_size=safe_bs2,
                                              shuffle=True, drop_last=True)
                    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)),
                                              shuffle=False)

                    # Train
                    try:
                        model = _train_from_images(
                            model, None, None, None, None, cfg, device,
                        ) if False else model  # placeholder, use inline training below

                        model = model.to(device)
                        optimizer = optim.Adam(model.parameters(),
                                               lr=cfg["training"]["learning_rate"],
                                               weight_decay=0.01)
                        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, 'min', patience=3, factor=0.5)
                        criterion = nn.CrossEntropyLoss()

                        patience_count = 10
                        trigger_times = 0
                        best_val_acc = 0
                        epochs = cfg["training"]["epochs"]

                        use_amp = (device == "cuda")
                        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

                        for epoch in range(epochs):
                            model.train()
                            running_loss, correct, total = 0.0, 0, 0
                            for images, labels in train_loader:
                                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                                optimizer.zero_grad(set_to_none=True)
                                with torch.amp.autocast("cuda", enabled=use_amp):
                                    outputs = model(images)
                                    loss = criterion(outputs, labels)
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()
                                running_loss += loss.item()
                                _, predicted = outputs.max(1)
                                total += labels.size(0)
                                correct += predicted.eq(labels).sum().item()

                            train_acc = 100.0 * correct / total

                            model.eval()
                            val_loss, vc, vt = 0.0, 0, 0
                            with torch.amp.autocast("cuda", enabled=use_amp):
                                with torch.no_grad():
                                    for images, labels in val_loader:
                                        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                                        outputs = model(images)
                                        loss = criterion(outputs, labels)
                                        val_loss += loss.item()
                                        _, predicted = outputs.max(1)
                                        vt += labels.size(0)
                                        vc += predicted.eq(labels).sum().item()

                            val_acc = 100.0 * vc / vt if vt > 0 else 0
                            vl = val_loss / max(len(val_loader), 1)

                            print(f"[Epoch {epoch+1}] Train: {train_acc:.1f}%, "
                                  f"Val: {val_acc:.1f}%")

                            scheduler.step(vl)
                            if val_acc > best_val_acc:
                                best_val_acc = val_acc
                                trigger_times = 0
                            else:
                                trigger_times += 1
                                if trigger_times >= patience_count:
                                    print("Early stopping.")
                                    break

                        # Evaluate on ORIGINAL test set (not augmented)
                        model.eval()
                        test_config = _build_run_config(cfg, dataset_entry, chart_type)
                        _ensure_base_images(test_config)
                        test_loaders = create_dataloaders(test_config)
                        test_loader = test_loaders["test"]["chart"]

                        clean_acc = _evaluate_accuracy(
                            model, test_loader, device, augment_fn=None)

                    except Exception as e:
                        print(f"    FAILED: {e}")
                        import traceback
                        traceback.print_exc()
                        clean_acc = -1

                    print(f"    Clean accuracy: {clean_acc:.4f}")
                    accuracy_rows.append({
                        "seed": seed, "dataset": dataset_name,
                        "multiplier": multiplier, "chart_type": chart_type,
                        "train_size": len(X_aug), "accuracy": clean_acc,
                    })
                    # Log clean accuracy to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/aug{multiplier}x/{chart_type}/s{seed}",
                        config={"multiplier": multiplier, "chart_type": chart_type,
                                "train_size": len(X_aug), "seed": seed,
                                "dataset": dataset_name},
                        accuracy=clean_acc,
                    )

                    del model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    fieldnames = ["seed", "dataset", "multiplier", "chart_type", "train_size", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — Accuracy vs Augmentation Multiplier")
    print(f"{'=' * 60}")
    from collections import defaultdict
    acc_map = defaultdict(list)
    for row in accuracy_rows:
        if row["accuracy"] >= 0:
            key = (row["dataset"], row["multiplier"])
            acc_map[key].append(row["accuracy"])

    all_ds = sorted(set(ds for (ds, _) in acc_map.keys()))
    for ds in all_ds:
        print(f"\n  {ds}:")
        for mult in multipliers:
            accs = acc_map.get((ds, mult), [])
            if accs:
                label = "original" if mult == 0 else f"{mult}x augmented"
                print(f"    {label:>16s}:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    hb.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 5K: TS Augmentation")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
