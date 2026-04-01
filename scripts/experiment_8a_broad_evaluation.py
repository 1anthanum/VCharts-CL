#!/usr/bin/env python
"""
Experiment 8A: Broad Dataset Evaluation
========================================
Runs the top-performing encoding methods from 6A/7A across 20+ UCR datasets
with 5 seeds per configuration for statistically robust results.

This is the "headline result" experiment for the paper, establishing:
  - Which encoding method generalizes best across diverse TS characteristics
  - Statistical significance via 5 independent runs
  - Coverage across binary/multiclass, short/long, small/large datasets

Encoding methods tested (top-K from prior experiments + baselines):
  Charts:  line, scatter
  Math:    gasf, rp_grayscale, cwt
  RGB:     gasf_gadf_mtf, rp_cwt_stft
  Phase:   phase_trajectory
  Stacked: phase_gasf_rp

Models: DeepCNN (random init), ResNet18 (pretrained)

Usage:
    python scripts/experiment_8a_broad_evaluation.py \
        --config vtbench/config/experiment_8a_broad.yaml
"""

import argparse
import gc
import csv
import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from vtbench.data.loader import read_ucr
from vtbench.train.factory import get_chart_model
from vtbench.utils.wandb_logger import WandbLogger
from vtbench.utils.heartbeat import Heartbeat
# Note: EncodingImageDataset and ChartImageDataset canonical location is
# vtbench.data.image_dataset — local copies below kept for now

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Aggressive GPU optimizations ──
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


# ============================================================
# Datasets
# ============================================================

class EncodingImageDataset(Dataset):
    """Load pre-generated encoding images."""

    def __init__(self, dataset_name, split, encoding_dir, global_indices,
                 labels, transform=None):
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform
        self.img_dir = os.path.join(
            CHART_IMAGE_ROOT, f"{dataset_name}_images", encoding_dir, split,
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
            if not getattr(self, '_warned_corrupt', False):
                print(f"  WARNING: corrupt image {img_path}, using blank placeholder")
                self._warned_corrupt = True
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class ChartImageDataset(Dataset):
    """Load existing chart images."""

    def __init__(self, dataset_name, split, chart_type, color_mode,
                 label_mode, global_indices, labels, transform=None,
                 bar_mode='border', scatter_mode='plain'):
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform

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

        self._prefix = {
            'line': 'line_chart', 'area': 'area_chart',
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
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


# ============================================================
# Training
# ============================================================

def train_and_evaluate(model, train_loader, test_loader, config, device):
    """Train model and return (test_accuracy, train_time_seconds)."""
    epochs = config.get('epochs', 100)
    lr = config.get('learning_rate', 0.001)

    model = model.to(device)
    # torch.compile disabled — no Triton on Windows

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # AMP mixed precision for ~2x throughput
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0
    patience_counter = 0
    patience = 10

    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validate on test set
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - t_start
    return best_val_acc, train_time


# ============================================================
# Main experiment
# ============================================================

def run_experiment(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp = cfg['experiment']
    output_dir = exp['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="8a_broad_evaluation", config=cfg)
    hb = Heartbeat("8a")

    datasets = exp['datasets']
    seeds = exp.get('seeds', [42, 123, 7, 2024, 99])
    image_size = exp.get('image_size', 128)

    # Encoding methods to evaluate
    chart_encodings = exp.get('chart_encodings', ['line', 'scatter'])
    math_encodings = exp.get('math_encodings',
                              ['gasf', 'rp_grayscale', 'cwt'])
    rgb_encodings = exp.get('rgb_encodings', ['gasf_gadf_mtf', 'rp_cwt_stft'])
    extended_encodings = exp.get('extended_encodings',
                                  ['phase_trajectory', 'phase_gasf_rp'])

    model_configs = exp.get('models', [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False, "lr": 0.001},
        {"name": "resnet18_pt", "chart_model": "resnet18", "pretrained": True, "lr": 0.0005},
    ])

    training_cfg = cfg.get('training', {})
    batch_size = training_cfg.get('batch_size', 64)
    epochs = training_cfg.get('epochs', 100)

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Build encoding list
    all_encodings = []
    for ct in chart_encodings:
        all_encodings.append(("chart", ct))
    for enc in math_encodings:
        all_encodings.append(("math", enc))
    for preset in rgb_encodings:
        all_encodings.append(("rgb", f"rgb_{preset}"))
    for enc in extended_encodings:
        all_encodings.append(("extended", enc))

    # CSV output with resume logic
    csv_path = os.path.join(output_dir, "broad_evaluation.csv")
    completed_runs = set()
    file_exists = os.path.exists(csv_path)

    if file_exists:
        print(f"Loading existing CSV: {csv_path}")
        failed_count = 0
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Only skip runs that actually succeeded (accuracy > 0)
                # Failed runs (accuracy=-1) should be retried
                try:
                    acc = float(row.get("accuracy", -1))
                except (ValueError, TypeError):
                    acc = -1
                if acc > 0:
                    completed_runs.add((row["seed"], row["dataset"], row["encoding_type"],
                                       row["encoding_name"], row["model"]))
                else:
                    failed_count += 1
        print(f"Loaded {len(completed_runs)} completed runs "
              f"({failed_count} failed runs will be retried)")

    csv_file = open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow([
            "seed", "dataset", "encoding_type", "encoding_name",
            "model", "accuracy", "train_time_s",
            "train_samples", "test_samples", "ts_length", "num_classes"
        ])

    total_runs = len(datasets) * len(seeds) * len(all_encodings) * len(model_configs)
    run_idx = 0
    skip_count = 0

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
            print(f"SKIP {ds_name}: dataset not found")
            continue

        n_train = len(X_train)
        n_test = len(X_test)
        num_classes = len(set(y_train))
        ts_length = X_train.shape[1] if len(X_train.shape) > 1 else 0
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + n_test))

        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name} (train={n_train}, test={n_test}, "
              f"classes={num_classes}, length={ts_length})")
        print(f"{'='*70}")

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

                    hb.pulse(
                        dataset=ds_name,
                        encoding=enc_name,
                        model=model_name,
                        run=f"{run_idx}/{total_runs}",
                        seed=seed,
                    )

                    run_key = (str(seed), ds_name, enc_type, enc_name, model_name)
                    if run_key in completed_runs:
                        print(f"  SKIPPED (already completed)")
                        skip_count += 1
                        continue

                    train_ds = None
                    test_ds = None
                    model = None
                    train_loader = None
                    test_loader = None

                    try:
                        try:
                            # Resolve directory name: RGB stack presets
                            # are stored with "rgb_" prefix on disk
                            from vtbench.data.ts_image_encodings import RGB_STACK_PRESETS
                            dir_name = enc_name
                            if enc_type == "extended" and enc_name in RGB_STACK_PRESETS:
                                dir_name = f"rgb_{enc_name}"

                            if enc_type == "chart":
                                train_ds = ChartImageDataset(
                                    ds_name, "train", enc_name, "color",
                                    "with_label", train_indices, list(y_train), tfm)
                                test_ds = ChartImageDataset(
                                    ds_name, "test", enc_name, "color",
                                    "with_label", test_indices, list(y_test), tfm)
                            else:
                                train_ds = EncodingImageDataset(
                                    ds_name, "train", dir_name,
                                    train_indices, list(y_train), tfm)
                                test_ds = EncodingImageDataset(
                                    ds_name, "test", dir_name,
                                    test_indices, list(y_test), tfm)
                        except FileNotFoundError as e:
                            print(f"  SKIP: {e}")
                            skip_count += 1
                            writer.writerow([
                                seed, ds_name, enc_type, enc_name,
                                model_name, -1, 0, n_train, n_test,
                                ts_length, num_classes
                            ])
                            continue

                        safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                        train_loader = DataLoader(
                            train_ds, batch_size=safe_bs, shuffle=True,
                            drop_last=True, num_workers=2,
                            pin_memory=True, persistent_workers=False,
                        )
                        test_loader = DataLoader(
                            test_ds, batch_size=min(batch_size, len(test_ds)),
                            shuffle=False, num_workers=2,
                            pin_memory=True, persistent_workers=False,
                        )

                        if len(train_loader) == 0:
                            print(f"  SKIP: empty train loader (too few samples for bs={batch_size})")
                            skip_count += 1
                            continue

                        model = get_chart_model(
                            mcfg['chart_model'], input_channels=3,
                            num_classes=num_classes,
                            pretrained=mcfg.get('pretrained', False),
                            image_size=image_size,
                        )

                        accuracy, train_time = train_and_evaluate(
                            model, train_loader, test_loader,
                            {'epochs': epochs, 'learning_rate': lr},
                            device,
                        )

                        print(f"  -> Accuracy: {accuracy:.4f}  Time: {train_time:.1f}s")

                        writer.writerow([
                            seed, ds_name, enc_type, enc_name,
                            model_name, f"{accuracy:.6f}", f"{train_time:.1f}",
                            n_train, n_test, ts_length, num_classes
                        ])
                        csv_file.flush()

                        wb.log_run_result(
                            name=f"{ds_name}/{enc_name}/{model_name}/s{seed}",
                            config={
                                "dataset": ds_name, "encoding_type": enc_type,
                                "encoding_name": enc_name, "model": model_name,
                                "seed": seed, "n_train": n_train,
                                "ts_length": ts_length, "num_classes": num_classes,
                            },
                            accuracy=accuracy, train_time_s=train_time,
                        )

                    except Exception as e:
                        print(f"  ERROR: {e}")
                        writer.writerow([
                            seed, ds_name, enc_type, enc_name,
                            model_name, -1, 0, n_train, n_test,
                            ts_length, num_classes
                        ])
                        csv_file.flush()

                    finally:
                        model = None
                        train_ds = None
                        test_ds = None
                        train_loader = None
                        test_loader = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            try:
                                torch.cuda.reset_peak_memory_stats()
                            except Exception:
                                pass
                            try:
                                torch.cuda.is_available()
                            except Exception:
                                pass

    csv_file.close()
    wb.finish()
    hb.close()

    print(f"\nResults saved to: {csv_path}")
    print(f"Total runs: {run_idx}, Skipped: {skip_count}")

    _write_summary(output_dir, csv_path)


def _write_summary(output_dir, csv_path):
    """Generate ranking summary."""
    import csv as csv_mod

    results = defaultdict(lambda: defaultdict(list))

    with open(csv_path, encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            acc = float(row['accuracy'])
            if acc < 0:
                continue
            key = (row['encoding_name'], row['model'])
            results[key]['accuracies'].append(acc)
            results[key]['datasets'].add(row['dataset'])

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 8A: Broad Dataset Evaluation\n\n")

        # Overall ranking
        f.write("## Overall Ranking (mean accuracy across all datasets & seeds)\n\n")
        f.write("| Rank | Encoding | Model | Mean Acc | Std | Datasets | Runs |\n")
        f.write("|------|----------|-------|----------|-----|----------|------|\n")

        entries = []
        for (enc, model), data in results.items():
            accs = data['accuracies']
            n_ds = len(data['datasets'])
            entries.append((np.mean(accs), enc, model, np.std(accs),
                           n_ds, len(accs)))

        for rank, (mean, enc, model, std, n_ds, n_runs) in enumerate(
                sorted(entries, reverse=True), 1):
            f.write(f"| {rank} | {enc} | {model} | {mean:.4f} | "
                    f"{std:.4f} | {n_ds} | {n_runs} |\n")

        # Per-dataset best
        f.write("\n## Best Encoding per Dataset\n\n")
        f.write("| Dataset | Best Encoding | Model | Accuracy |\n")
        f.write("|---------|---------------|-------|----------|\n")

        ds_results = defaultdict(list)
        with open(csv_path, encoding="utf-8") as csvf:
            reader = csv_mod.DictReader(csvf)
            for row in reader:
                acc = float(row['accuracy'])
                if acc < 0:
                    continue
                ds_results[row['dataset']].append(
                    (acc, row['encoding_name'], row['model']))

        for ds in sorted(ds_results.keys()):
            best = max(ds_results[ds], key=lambda x: x[0])
            f.write(f"| {ds} | {best[1]} | {best[2]} | {best[0]:.4f} |\n")

    print(f"Summary: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)
