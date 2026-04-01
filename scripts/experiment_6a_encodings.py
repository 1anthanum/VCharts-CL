#!/usr/bin/env python
"""
Experiment 6A: Image Encoding Method Comparison
=================================================
Compares ALL image encoding methods head-to-head:
  - Chart-based: line, area, bar, scatter
  - Mathematical: GASF, GADF, MTF, RP, RP_grayscale, CWT, STFT
  - RGB stacked: gasf_gadf_mtf, rp_cwt_stft
  - Colormapped: gasf_viridis, gadf_plasma, rp_grayscale_inferno

Uses DeepCNN (unpretrained) and ResNet18 (pretrained) on multiple datasets.
This is the MOST IMPORTANT experiment for the paper — it directly answers
"which visual representation works best for TSC?"

Usage:
    python scripts/experiment_6a_encodings.py --config vtbench/config/experiment_6a_encodings.yaml
"""

import argparse
import csv
import gc
import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from vtbench.data.loader import read_ucr
from vtbench.train.factory import get_chart_model
from vtbench.utils.wandb_logger import WandbLogger
# Note: EncodingImageDataset and ChartImageDataset canonical location is
# vtbench.data.image_dataset — local copies below kept for now

device = "cuda" if torch.cuda.is_available() else "cpu"

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


# ============================================================
# Dataset that loads pre-generated encoding images
# ============================================================

class EncodingImageDataset(Dataset):
    """Load pre-generated encoding images from disk."""

    def __init__(self, dataset_name, split, encoding_dir, global_indices,
                 labels, transform=None):
        self.dataset_name = dataset_name
        self.split = split
        self.encoding_dir = encoding_dir
        self.global_indices = global_indices
        self.labels = labels
        self.transform = transform

        # Verify directory exists
        self.img_dir = os.path.join(
            CHART_IMAGE_ROOT,
            f"{dataset_name}_images",
            encoding_dir,
            split,
        )
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Encoding images not found: {self.img_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        img_path = os.path.join(self.img_dir, f"sample_{gidx}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Corrupt image — regenerate on the fly
            img = self._regenerate(img_path, idx)
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label

    def _regenerate(self, img_path, idx):
        """Regenerate a corrupt or missing encoding image."""
        from vtbench.data.ts_image_encodings import (
            get_encoding, get_rgb_stack, apply_colormap,
            save_encoding_image, ENCODING_REGISTRY, RGB_STACK_PRESETS,
        )
        from vtbench.data.loader import read_ucr

        enc = self.encoding_dir
        # Load the time series for this sample
        ds = self.dataset_name
        for ext in [".tsv", ".ts"]:
            p = f"UCRArchive_2018/{ds}/{ds}_{'TRAIN' if self.split == 'train' else 'TEST'}.{ext}"
            if os.path.exists(p):
                X, _ = read_ucr(p)
                break
        else:
            raise FileNotFoundError(f"Cannot find dataset {ds} to regenerate image")

        local_idx = idx  # index within the split
        ts = X[local_idx]

        is_rgb = enc.startswith("rgb_")
        is_colormap = any(enc.endswith(f"_{cm}") for cm in
                          ["viridis", "plasma", "inferno", "magma", "jet"])

        if is_rgb:
            arr = get_rgb_stack(enc.replace("rgb_", ""), ts, 128)
        elif is_colormap:
            base, cmap = enc.rsplit("_", 1)
            gray = get_encoding(base, ts, 128)
            arr = apply_colormap(gray, cmap)
        else:
            arr = get_encoding(enc, ts, 128)

        save_encoding_image(arr, img_path)
        return Image.open(img_path).convert("RGB")


class ChartImageDataset(Dataset):
    """Load existing chart images (line/area/bar/scatter)."""

    def __init__(self, dataset_name, split, chart_type, color_mode,
                 label_mode, global_indices, labels, transform=None,
                 bar_mode='border', scatter_mode='plain'):
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform
        self.chart_type = chart_type

        # Build directory path matching chart_generator.py conventions
        base = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images")
        if chart_type == 'bar':
            self.img_dir = os.path.join(
                base, f"bar_charts_{bar_mode}_{color_mode}_{label_mode}", split)
        elif chart_type == 'scatter':
            self.img_dir = os.path.join(
                base, f"scatter_charts_{scatter_mode}_{color_mode}_{label_mode}", split)
        else:
            self.img_dir = os.path.join(
                base, f"{chart_type}_charts_{color_mode}_{label_mode}", split)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Chart images not found: {self.img_dir}")

        # Detect filename pattern
        self._prefix = {
            'line': 'line_chart',
            'area': 'area_chart',
            'bar': f'bar_chart_{bar_mode}',
            'scatter': f'scatter_chart_{scatter_mode}',
        }[chart_type]
        self._suffix = f"_{color_mode}_{label_mode}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        fname = f"{self._prefix}{self._suffix}_{gidx}.png"
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ============================================================
# Training loop
# ============================================================

def train_and_evaluate(model, train_loader, test_loader, config, device):
    """Train a model and return test accuracy."""
    epochs = config.get('epochs', 100)
    lr = config.get('learning_rate', 0.001)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total if total > 0 else 0

        # Validate on test set
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_loss)

        print(f"  [Epoch {epoch+1}] Train Acc: {train_acc:.2%}, "
              f"Val Acc: {val_acc:.2%}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping.")
                break

    # Move model off GPU before returning to free VRAM
    model.cpu()
    return best_val_acc


# ============================================================
# Main experiment
# ============================================================

def run_experiment(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp = cfg['experiment']
    output_dir = exp['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="6a_encodings", config=cfg)

    datasets = exp['datasets']
    seeds = exp.get('seeds', [42, 123, 7])
    image_size = exp.get('image_size', 128)

    # Encoding methods to test
    chart_encodings = exp.get('chart_encodings', ['line', 'area', 'bar', 'scatter'])
    math_encodings = exp.get('math_encodings',
                              ['gasf', 'gadf', 'mtf', 'rp', 'rp_grayscale', 'cwt', 'stft'])
    rgb_encodings = exp.get('rgb_encodings', ['gasf_gadf_mtf', 'rp_cwt_stft'])
    colormap_encodings = exp.get('colormap_encodings',
                                  ['gasf_viridis', 'gadf_plasma', 'rp_grayscale_inferno'])

    # Models to test
    model_configs = exp.get('models', [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False, "lr": 0.001},
        {"name": "resnet18_pt", "chart_model": "resnet18", "pretrained": True, "lr": 0.0005},
    ])

    training_cfg = cfg.get('training', {})
    batch_size = training_cfg.get('batch_size', 64)
    epochs = training_cfg.get('epochs', 100)

    # Image transforms
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # CSV output — resume-aware: load existing results, skip completed runs
    # Only skip runs that actually succeeded (accuracy > 0).
    # Failed runs (accuracy <= 0) will be retried.
    csv_path = os.path.join(output_dir, "encoding_comparison.csv")
    completed_runs = set()
    file_exists = os.path.exists(csv_path)

    if file_exists:
        failed_count = 0
        with open(csv_path, encoding="utf-8") as f_exist:
            reader = csv.DictReader(f_exist)
            for row in reader:
                try:
                    acc = float(row.get("accuracy", -1))
                except (ValueError, TypeError):
                    acc = -1
                if acc > 0:
                    key = (row['seed'], row['dataset'], row['encoding_name'], row['model'])
                    completed_runs.add(key)
                else:
                    failed_count += 1
        print(f"Resuming: {len(completed_runs)} completed runs "
              f"({failed_count} failed runs will be retried), appending.")
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
    else:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(csv_file)
        writer.writerow([
            "seed", "dataset", "encoding_type", "encoding_name",
            "model", "accuracy", "train_samples", "test_samples", "num_classes"
        ])

    all_encodings = []

    # Chart-based encodings
    for ct in chart_encodings:
        all_encodings.append(("chart", ct))

    # Mathematical encodings
    for enc in math_encodings:
        all_encodings.append(("math", enc))

    # RGB stacks
    for preset in rgb_encodings:
        all_encodings.append(("rgb", f"rgb_{preset}"))

    # Colormapped
    for cm_enc in colormap_encodings:
        all_encodings.append(("colormap", cm_enc))

    total_runs = len(datasets) * len(seeds) * len(all_encodings) * len(model_configs)
    run_idx = 0

    for ds_name in datasets:
        # Load data
        X_train, y_train, X_test, y_test = None, None, None, None
        for ext in [".tsv", ".ts"]:
            train_path = f"UCRArchive_2018/{ds_name}/{ds_name}_TRAIN{ext}"
            test_path = f"UCRArchive_2018/{ds_name}/{ds_name}_TEST{ext}"
            if os.path.exists(train_path):
                X_train, y_train = read_ucr(train_path)
                X_test, y_test = read_ucr(test_path)
                break

        if X_train is None:
            print(f"SKIP {ds_name}: not found")
            continue

        n_train = len(X_train)
        n_test = len(X_test)
        num_classes = len(set(y_train))
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + n_test))

        print(f"\nDataset: {ds_name} ({n_train} train, {n_test} test, "
              f"{num_classes} classes)")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            for enc_type, enc_name in all_encodings:
                for mcfg in model_configs:
                    run_idx += 1
                    model_name = mcfg['name']
                    lr = mcfg.get('lr', 0.001)

                    print(f"\n[{run_idx}/{total_runs}] {ds_name} | "
                          f"{enc_name} | {model_name} | seed={seed}")

                    # Skip if already completed (resume mode)
                    run_key = (str(seed), ds_name, enc_name, model_name)
                    if run_key in completed_runs:
                        print(f"  SKIP (already completed)")
                        continue

                    # Wrap entire run in try/except for CUDA error recovery
                    try:
                        # Create dataset
                        try:
                            if enc_type == "chart":
                                train_ds = ChartImageDataset(
                                    ds_name, "train", enc_name, "color",
                                    "with_label", train_indices, list(y_train), tfm
                                )
                                test_ds = ChartImageDataset(
                                    ds_name, "test", enc_name, "color",
                                    "with_label", test_indices, list(y_test), tfm
                                )
                            else:
                                train_ds = EncodingImageDataset(
                                    ds_name, "train", enc_name,
                                    train_indices, list(y_train), tfm
                                )
                                test_ds = EncodingImageDataset(
                                    ds_name, "test", enc_name,
                                    test_indices, list(y_test), tfm
                                )
                        except FileNotFoundError as e:
                            print(f"  SKIP: {e}")
                            writer.writerow([
                                seed, ds_name, enc_type, enc_name,
                                model_name, -1, n_train, n_test, num_classes
                            ])
                            continue

                        safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                        train_loader = DataLoader(
                            train_ds, batch_size=safe_bs, shuffle=True,
                            drop_last=True, num_workers=0,
                        )
                        test_loader = DataLoader(
                            test_ds, batch_size=min(batch_size, len(test_ds)),
                            shuffle=False, num_workers=0,
                        )

                        if len(train_loader) == 0:
                            print(f"  SKIP: empty train loader")
                            continue

                        # Create model
                        model = get_chart_model(
                            mcfg['chart_model'],
                            input_channels=3,
                            num_classes=num_classes,
                            pretrained=mcfg.get('pretrained', False),
                            image_size=image_size,
                        )

                        # Train and evaluate
                        accuracy = train_and_evaluate(
                            model, train_loader, test_loader,
                            {'epochs': epochs, 'learning_rate': lr},
                            device,
                        )

                        print(f"  -> Accuracy: {accuracy:.4f}")

                        writer.writerow([
                            seed, ds_name, enc_type, enc_name,
                            model_name, f"{accuracy:.6f}",
                            n_train, n_test, num_classes
                        ])
                        csv_file.flush()

                        # Log to W&B
                        run_info = {
                            "dataset": ds_name,
                            "encoding_type": enc_type,
                            "encoding_name": enc_name,
                            "model": model_name,
                            "seed": seed,
                        }
                        wb.log_run_result(
                            name=f"{ds_name}_{enc_name}_{model_name}_seed{seed}",
                            config=run_info,
                            accuracy=accuracy,
                        )

                    except Exception as cuda_err:
                        # CUDA / runtime error recovery: log failure, continue
                        print(f"  RUN FAILED: {cuda_err}")
                        writer.writerow([
                            seed, ds_name, enc_type, enc_name,
                            model_name, -1, n_train, n_test, num_classes
                        ])
                        csv_file.flush()

                    finally:
                        # Aggressive GPU cleanup between ALL runs
                        model = None
                        train_loader = None
                        test_loader = None
                        train_ds = None
                        test_ds = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    csv_file.close()
    print(f"\nResults saved to: {csv_path}")

    # Write summary
    _write_summary(output_dir, csv_path)

    # Finish W&B logging
    wb.finish()


def _write_summary(output_dir, csv_path):
    """Generate summary markdown."""
    import csv as csv_mod
    from collections import defaultdict

    results = defaultdict(lambda: defaultdict(list))

    with open(csv_path, encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            acc = float(row['accuracy'])
            if acc < 0:
                continue
            key = (row['encoding_name'], row['model'])
            results[row['dataset']][key].append(acc)

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 6A: Image Encoding Comparison\n\n")

        for ds in sorted(results.keys()):
            f.write(f"## {ds}\n\n")
            f.write("| Encoding | Model | Mean Acc | Std | Min | Max |\n")
            f.write("|----------|-------|----------|-----|-----|-----|\n")

            entries = []
            for (enc, model), accs in results[ds].items():
                mean = np.mean(accs)
                std = np.std(accs)
                entries.append((mean, enc, model, std, min(accs), max(accs)))

            for mean, enc, model, std, mn, mx in sorted(entries, reverse=True):
                f.write(f"| {enc} | {model} | {mean:.4f} | {std:.4f} | "
                        f"{mn:.4f} | {mx:.4f} |\n")
            f.write("\n")

    print(f"Summary: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)
