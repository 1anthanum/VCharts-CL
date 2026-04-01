#!/usr/bin/env python
"""
Experiment 7C: Chart Visual Ablation + Extended Augmentation
============================================================
Two-part experiment:

Part 1 - Chart decoration ablation:
  Tests whether chart decorations (axes, labels, grid) help or hurt
  CNN classification. Compares:
    - with_label:    full matplotlib chart (axes + ticks + title)
    - without_label: pure signal fill (no axes/ticks/margins)

Part 2 - Extended augmentation techniques:
  Tests additional image augmentation strategies beyond the standard
  set (gaussian blur/noise/crop/jpeg). New augmentations:
    - Color jitter (brightness, contrast, saturation)
    - Random erasing (cutout)
    - Random rotation (small angles)
    - Random perspective
    - Mixup at image level
    - CutMix

Usage
-----
    python scripts/experiment_7c_chart_ablation.py \\
        --config vtbench/config/experiment_7c_chart_ablation.yaml
"""

import argparse
import csv
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageEnhance
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtbench.data.loader import read_ucr
from vtbench.data.chart_generator import TimeSeriesImageDataset
from vtbench.train.factory import get_chart_model
from vtbench.utils.wandb_logger import WandbLogger
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


# ====================================================================
# Extended Augmentations (torchvision-based, applied during training)
# ====================================================================

def get_augmented_transform(aug_name, image_size=128):
    """Return a torchvision transform chain for the given augmentation."""
    base = [
        transforms.Resize((image_size, image_size)),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    aug_map = {
        'none': base + normalize,

        'color_jitter': base + [
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
        ] + normalize,

        'random_erasing': base + normalize + [
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.15)),
        ],

        'rotation_5': base + [
            transforms.RandomRotation(5),
        ] + normalize,

        'rotation_15': base + [
            transforms.RandomRotation(15),
        ] + normalize,

        'perspective': base + [
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ] + normalize,

        'horizontal_flip': base + [
            transforms.RandomHorizontalFlip(p=0.5),
        ] + normalize,

        'auto_augment': base + [
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
        ] + normalize,

        'trivial_augment': base + [
            transforms.TrivialAugmentWide(),
        ] + normalize,

        'combined_light': base + [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(3),
        ] + normalize,

        'combined_heavy': base + [
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2),
            transforms.RandomRotation(10),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
        ] + normalize + [
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ],
    }

    if aug_name not in aug_map:
        raise ValueError(f"Unknown augmentation: {aug_name}. "
                         f"Available: {list(aug_map.keys())}")

    return transforms.Compose(aug_map[aug_name])


# ====================================================================
# Mixup & CutMix (batch-level augmentation)
# ====================================================================

def mixup_data(x, y, alpha=0.2):
    """Apply Mixup to a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    _, _, h, w = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    return mixed_x, y, y[index], lam


# ====================================================================
# Chart Image Dataset with flexible augmentation
# ====================================================================

class ChartDatasetAug(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels, global_indices, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def _find_image(self, g_idx):
        # Try full prefix patterns (e.g. line_chart_color_with_label_0.png)
        for prefix in ['line_chart_color_with_label',
                        'line_chart_color_without_label',
                        'area_chart_color_with_label',
                        'area_chart_color_without_label',
                        'bar_chart_border_color_with_label',
                        'bar_chart_border_color_without_label',
                        'scatter_chart_plain_color_with_label',
                        'scatter_chart_plain_color_without_label',
                        # Legacy short prefixes
                        'line_chart', 'area_chart', 'bar_chart',
                        'scatter_chart', 'sample']:
            path = os.path.join(self.image_dir, f"{prefix}_{g_idx}.png")
            if os.path.exists(path):
                return path
        return None

    def __getitem__(self, idx):
        g_idx = self.global_indices[idx]
        path = self._find_image(g_idx)
        if path is None:
            # Warn loudly instead of silently returning black image
            if not hasattr(self, '_warned'):
                self._warned = True
                print(f"  WARNING: image not found for index {g_idx} in {self.image_dir}")
                print(f"  Available files sample: {os.listdir(self.image_dir)[:3]}")
            tensor = torch.zeros(3, 128, 128)
            return tensor, self.labels[idx]

        img = Image.open(path).convert("RGB")
        if self.transform:
            tensor = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
        return tensor, self.labels[idx]


# ====================================================================
# Training with optional batch-level augmentation
# ====================================================================

def train_and_evaluate(train_ds, test_ds, model_name, pretrained,
                       lr, num_classes, epochs=100, batch_size=32,
                       batch_aug=None):
    """
    batch_aug: None, 'mixup', or 'cutmix'
    """
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    if len(train_loader) == 0:
        return 0.0

    model = get_chart_model(model_name, num_classes=num_classes,
                            pretrained=pretrained).to(device)
    # torch.compile disabled — no Triton on Windows
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            if batch_aug == 'mixup':
                images, y_a, y_b, lam = mixup_data(images, labels)
                logits = model(images)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            elif batch_aug == 'cutmix':
                images, y_a, y_b, lam = cutmix_data(images, labels)
                logits = model(images)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds = torch.argmax(model(images), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = correct / total if total > 0 else 0.0
        scheduler.step(total_loss / len(train_loader))

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break

    return best_acc


# ====================================================================
# Main
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_7c_chart_ablation")
    os.makedirs(output_dir, exist_ok=True)

    datasets = exp["datasets"]
    seeds = exp.get("seeds", [42, 123, 7])
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    epochs = cfg.get("training", {}).get("epochs", 100)
    image_size = exp.get("image_size", 128)

    chart_types = exp.get("chart_types", ["line", "area"])
    label_modes = exp.get("label_modes", ["with_label", "without_label"])

    # Augmentation strategies (applied during training)
    augmentations = exp.get("augmentations", [
        "none", "color_jitter", "random_erasing", "rotation_5",
        "perspective", "trivial_augment", "combined_light", "combined_heavy",
    ])
    batch_augmentations = exp.get("batch_augmentations", [None, "mixup", "cutmix"])

    models = exp.get("models", [
        {"name": "resnet18_pt", "chart_model": "resnet18", "pretrained": True, "lr": 0.0005},
    ])

    dataset_root = exp.get("dataset_root", "UCRArchive_2018")
    dataset_ext = exp.get("dataset_ext", "tsv")

    CHART_DIR_MAP = {
        'line': 'line_charts_color',
        'area': 'area_charts_color',
        'bar': 'bar_charts_border_color',
        'scatter': 'scatter_charts_plain_color',
    }

    wb = WandbLogger(project="vtbench", experiment="7c_chart_ablation", config=cfg)

    rows = []

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        train_path = os.path.join(dataset_root, dataset_name,
                                  f"{dataset_name}_TRAIN.{dataset_ext}")
        test_path = os.path.join(dataset_root, dataset_name,
                                 f"{dataset_name}_TEST.{dataset_ext}")
        if not os.path.exists(train_path):
            train_path = train_path.replace(f".{dataset_ext}", ".ts")
            test_path = test_path.replace(f".{dataset_ext}", ".ts")

        X_train, y_train = read_ucr(train_path)
        X_test, y_test = read_ucr(test_path)
        n_train = len(X_train)
        num_classes = len(set(y_train))
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + len(X_test)))

        for chart_type in chart_types:
            for label_mode in label_modes:
                chart_dir_name = f"{CHART_DIR_MAP[chart_type]}_{label_mode}"
                base_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images")
                train_dir = os.path.join(base_dir, chart_dir_name, "train")
                test_dir = os.path.join(base_dir, chart_dir_name, "test")

                # Generate if needed
                if not os.path.isdir(train_dir):
                    print(f"  Generating {chart_type}/{label_mode} images...")
                    for split, X_s, y_s, g_idx in [
                        ("train", X_train, y_train, train_indices),
                        ("test", X_test, y_test, test_indices),
                    ]:
                        TimeSeriesImageDataset(
                            time_series_data=list(X_s),
                            labels=list(y_s),
                            dataset_name=dataset_name,
                            split=split,
                            chart_type=chart_type,
                            color_mode='color',
                            label_mode=label_mode,
                            generate_images=True,
                            global_indices=g_idx,
                        )

                if not os.path.isdir(train_dir):
                    print(f"  SKIP: {train_dir} not found")
                    continue

                # Part 1: Label mode ablation (no aug)
                # Part 2: Augmentation sweep (with_label only to avoid multiplying)
                aug_list = ["none"] if label_mode == "without_label" else augmentations

                for aug_name in aug_list:
                    batch_aug_list = [None] if aug_name != "none" else batch_augmentations

                    for batch_aug in batch_aug_list:
                        combo_name = aug_name
                        if batch_aug:
                            combo_name = f"{aug_name}+{batch_aug}"

                        train_transform = get_augmented_transform(aug_name, image_size)
                        test_transform = get_augmented_transform('none', image_size)

                        for model_cfg in models:
                            for seed in seeds:
                                torch.manual_seed(seed)
                                np.random.seed(seed)

                                train_ds = ChartDatasetAug(
                                    train_dir, list(y_train), train_indices,
                                    transform=train_transform)
                                test_ds = ChartDatasetAug(
                                    test_dir, list(y_test), test_indices,
                                    transform=test_transform)

                                acc = train_and_evaluate(
                                    train_ds, test_ds,
                                    model_cfg["chart_model"],
                                    model_cfg["pretrained"],
                                    model_cfg["lr"], num_classes,
                                    epochs, batch_size,
                                    batch_aug=batch_aug)

                                tag = f"{chart_type}/{label_mode}/{combo_name}"
                                print(f"  {tag} {model_cfg['name']} s={seed}: {acc:.4f}")

                                run_info = {
                                    "dataset": dataset_name,
                                    "chart_type": chart_type,
                                    "label_mode": label_mode,
                                    "augmentation": combo_name,
                                    "model": model_cfg["name"],
                                    "seed": seed,
                                    "accuracy": f"{acc:.4f}",
                                }
                                rows.append(run_info)
                                wb.log_run_result(
                                    name=f"{dataset_name}/{chart_type}/{label_mode}/{combo_name}/s{seed}",
                                    config=run_info, accuracy=acc,
                                )

    # Save
    csv_path = os.path.join(output_dir, "accuracy_chart_ablation.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "chart_type", "label_mode",
                                                "augmentation", "model", "seed", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 7C: Chart Ablation + Extended Augmentation\n\n")

        # Part 1: Label mode comparison
        f.write("## Part 1: With Label vs Without Label\n\n")
        f.write("| Dataset | Chart | with_label | without_label | Delta |\n")
        f.write("|---|---|---|---|---|\n")
        for ds in datasets:
            for ct in chart_types:
                wl = [float(r["accuracy"]) for r in rows
                      if r["dataset"] == ds and r["chart_type"] == ct
                      and r["label_mode"] == "with_label" and r["augmentation"] == "none"]
                wol = [float(r["accuracy"]) for r in rows
                       if r["dataset"] == ds and r["chart_type"] == ct
                       and r["label_mode"] == "without_label" and r["augmentation"] == "none"]
                if wl and wol:
                    wl_m, wol_m = np.mean(wl), np.mean(wol)
                    f.write(f"| {ds} | {ct} | {wl_m:.4f} | {wol_m:.4f} | {wol_m-wl_m:+.4f} |\n")

        # Part 2: Augmentation ranking
        f.write("\n## Part 2: Augmentation Ranking (mean accuracy across datasets)\n\n")
        aug_scores = {}
        for r in rows:
            if r["label_mode"] == "with_label":
                key = r["augmentation"]
                aug_scores.setdefault(key, []).append(float(r["accuracy"]))
        sorted_augs = sorted(aug_scores.items(), key=lambda x: np.mean(x[1]), reverse=True)
        f.write("| Rank | Augmentation | Mean Acc | Std |\n")
        f.write("|---|---|---|---|\n")
        for i, (aug, accs) in enumerate(sorted_augs, 1):
            f.write(f"| {i} | {aug} | {np.mean(accs):.4f} | {np.std(accs):.4f} |\n")

    print(f"\nResults: {csv_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)
