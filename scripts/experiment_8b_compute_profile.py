#!/usr/bin/env python
"""
Experiment 8B: Computational Cost Profiling
=============================================
Measures training time, inference latency, peak GPU memory, and model
parameter count for every encoding method.

Produces a cost-benefit analysis table: accuracy vs. compute cost.
Runs on 2 representative datasets (one small, one large) with 1 seed.

Usage:
    python scripts/experiment_8b_compute_profile.py \
        --config vtbench/config/experiment_8b_compute.yaml
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

device = "cuda" if torch.cuda.is_available() else "cpu"

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


class EncodingImageDataset(Dataset):
    def __init__(self, dataset_name, split, encoding_dir, global_indices,
                 labels, transform=None):
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform
        self.img_dir = os.path.join(
            CHART_IMAGE_ROOT, f"{dataset_name}_images", encoding_dir, split)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Not found: {self.img_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        path = os.path.join(self.img_dir, f"sample_{gidx}.png")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


class ChartImageDataset(Dataset):
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
            raise FileNotFoundError(f"Not found: {self.img_dir}")
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
        path = os.path.join(self.img_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.long)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_latency(model, sample_input, n_warmup=10, n_runs=100):
    """Measure average inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            model(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return np.mean(times), np.std(times)


def train_and_profile(model, train_loader, test_loader, config, device):
    """Train model and return detailed profiling metrics."""
    epochs = config.get('epochs', 100)
    lr = config.get('learning_rate', 0.001)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0
    patience = 10
    epochs_trained = 0

    # Reset GPU memory tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    t_start = time.time()

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.no_grad():
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
        epochs_trained = epoch + 1

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    train_time = time.time() - t_start

    # Peak GPU memory
    peak_mem_mb = 0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Inference latency
    sample = next(iter(test_loader))[0][:1].to(device)
    inf_mean, inf_std = measure_inference_latency(model, sample)

    return {
        'accuracy': best_val_acc,
        'train_time_s': train_time,
        'epochs_trained': epochs_trained,
        'peak_gpu_mb': peak_mem_mb,
        'inference_ms_mean': inf_mean,
        'inference_ms_std': inf_std,
        'n_params': count_parameters(model),
    }


def run_experiment(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp = cfg['experiment']
    output_dir = exp['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="8b_compute_profile", config=cfg)
    hb = Heartbeat("8b")

    datasets = exp['datasets']
    seed = exp.get('seed', 42)
    image_size = exp.get('image_size', 128)

    # All encodings to profile
    chart_encodings = exp.get('chart_encodings', ['line', 'area', 'bar', 'scatter'])
    math_encodings = exp.get('math_encodings',
                              ['gasf', 'gadf', 'mtf', 'rp', 'rp_grayscale', 'cwt', 'stft'])
    rgb_encodings = exp.get('rgb_encodings', ['gasf_gadf_mtf', 'rp_cwt_stft'])
    extended_encodings = exp.get('extended_encodings', [
        'phase_trajectory', 'phase_scatter', 'phase_gasf_rp', 'phase_cwt_mtf',
    ])

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

    all_encodings = []
    for ct in chart_encodings:
        all_encodings.append(("chart", ct))
    for enc in math_encodings:
        all_encodings.append(("math", enc))
    for preset in rgb_encodings:
        all_encodings.append(("rgb", f"rgb_{preset}"))
    for enc in extended_encodings:
        all_encodings.append(("extended", enc))

    csv_path = os.path.join(output_dir, "compute_profile.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow([
        "dataset", "encoding_type", "encoding_name", "model",
        "accuracy", "train_time_s", "epochs_trained",
        "peak_gpu_mb", "inference_ms_mean", "inference_ms_std",
        "n_params", "train_samples", "test_samples", "num_classes",
    ])

    torch.manual_seed(seed)
    np.random.seed(seed)

    total = len(datasets) * len(all_encodings) * len(model_configs)
    run_idx = 0

    for ds_name in datasets:
        X_train, y_train, X_test, y_test = None, None, None, None
        for ext in [".tsv", ".ts"]:
            train_path = f"UCRArchive_2018/{ds_name}/{ds_name}_TRAIN{ext}"
            test_path = f"UCRArchive_2018/{ds_name}/{ds_name}_TEST{ext}"
            if os.path.exists(train_path):
                X_train, y_train = read_ucr(train_path)
                X_test, y_test = read_ucr(test_path)
                break
        if X_train is None:
            print(f"SKIP {ds_name}")
            continue

        n_train = len(X_train)
        n_test = len(X_test)
        num_classes = len(set(y_train))
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + n_test))

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} (train={n_train}, test={n_test})")
        print(f"{'='*60}")

        for enc_type, enc_name in all_encodings:
            for mcfg in model_configs:
                run_idx += 1
                model_name = mcfg['name']
                print(f"\n[{run_idx}/{total}] {ds_name} | {enc_name} | {model_name}")

                hb.pulse(
                    dataset=ds_name,
                    encoding=enc_name,
                    model=model_name,
                    run=f"{run_idx}/{total}",
                )

                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    if enc_type == "chart":
                        train_ds = ChartImageDataset(
                            ds_name, "train", enc_name, "color",
                            "with_label", train_indices, list(y_train), tfm)
                        test_ds = ChartImageDataset(
                            ds_name, "test", enc_name, "color",
                            "with_label", test_indices, list(y_test), tfm)
                    else:
                        train_ds = EncodingImageDataset(
                            ds_name, "train", enc_name,
                            train_indices, list(y_train), tfm)
                        test_ds = EncodingImageDataset(
                            ds_name, "test", enc_name,
                            test_indices, list(y_test), tfm)
                except FileNotFoundError as e:
                    print(f"  SKIP: {e}")
                    continue

                safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                train_loader = DataLoader(
                    train_ds, batch_size=safe_bs, shuffle=True,
                    drop_last=True)
                test_loader = DataLoader(
                    test_ds, batch_size=min(batch_size, len(test_ds)),
                    shuffle=False)

                if len(train_loader) == 0:
                    print(f"  SKIP: empty loader")
                    continue

                model = get_chart_model(
                    mcfg['chart_model'], input_channels=3,
                    num_classes=num_classes,
                    pretrained=mcfg.get('pretrained', False),
                    image_size=image_size)

                metrics = train_and_profile(
                    model, train_loader, test_loader,
                    {'epochs': epochs, 'learning_rate': mcfg.get('lr', 0.001)},
                    device)

                print(f"  Acc: {metrics['accuracy']:.4f}  "
                      f"Time: {metrics['train_time_s']:.1f}s  "
                      f"GPU: {metrics['peak_gpu_mb']:.0f}MB  "
                      f"Inf: {metrics['inference_ms_mean']:.2f}ms  "
                      f"Params: {metrics['n_params']:,}")

                writer.writerow([
                    ds_name, enc_type, enc_name, model_name,
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['train_time_s']:.1f}",
                    metrics['epochs_trained'],
                    f"{metrics['peak_gpu_mb']:.1f}",
                    f"{metrics['inference_ms_mean']:.3f}",
                    f"{metrics['inference_ms_std']:.3f}",
                    metrics['n_params'],
                    n_train, n_test, num_classes,
                ])
                csv_file.flush()

                wb.log_run_result(
                    name=f"{ds_name}/{enc_name}/{model_name}",
                    config={
                        "dataset": ds_name, "encoding": enc_name,
                        "model": model_name,
                    },
                    accuracy=metrics['accuracy'],
                    train_time_s=metrics['train_time_s'],
                    peak_gpu_mb=metrics['peak_gpu_mb'],
                    inference_ms=metrics['inference_ms_mean'],
                    n_params=metrics['n_params'],
                )

                del model
                gc.collect()
                torch.cuda.empty_cache()

    csv_file.close()
    wb.finish()
    hb.close()
    print(f"\nResults: {csv_path}")

    # Summary
    _write_summary(output_dir, csv_path)


def _write_summary(output_dir, csv_path):
    import csv as csv_mod
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            rows.append(row)

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 8B: Computational Cost Profile\n\n")
        f.write("| Encoding | Model | Accuracy | Train(s) | GPU(MB) | "
                "Inf(ms) | Params |\n")
        f.write("|----------|-------|----------|----------|---------|"
                "---------|--------|\n")

        for row in sorted(rows, key=lambda r: -float(r.get('accuracy', 0))):
            f.write(f"| {row['encoding_name']} | {row['model']} | "
                    f"{row['accuracy']} | {row['train_time_s']} | "
                    f"{row['peak_gpu_mb']} | {row['inference_ms_mean']} | "
                    f"{row['n_params']} |\n")

    print(f"Summary: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_experiment(args.config)
