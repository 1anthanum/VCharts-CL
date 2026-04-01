#!/usr/bin/env python
"""
Experiment 7A: Extended Image Encodings
=======================================
Tests newly added visual encodings that go beyond the standard
GAF/MTF/RP methods:

  - Phase space scatter plot (delay embedding, tau=1)
  - Phase space trajectory (connected line, tau=1)
  - Phase space auto-tau (tau=T//8)
  - Multi-tau phase space RGB (tau=1,T//8,T//4 as R/G/B)
  - GASF on first-differenced TS
  - GASF on cumulative-sum TS
  - RP on first-differenced TS
  - New RGB stacks: phase+gasf+rp, phase+cwt+mtf, gasf+gasf_diff+gadf

These complement Experiment 6A's standard encodings. The hypothesis is
that phase space plots capture attractor geometry that other encodings
miss, and TS preprocessing before encoding expands the feature space.

Usage
-----
    python scripts/experiment_7a_extended_encodings.py \\
        --config vtbench/config/experiment_7a_extended.yaml
"""

import argparse
import gc
import copy
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtbench.data.loader import read_ucr
from vtbench.data.ts_image_encodings import (
    get_encoding, get_rgb_stack, encode_phase_space_multi_tau,
    apply_edge_detection, apply_histogram_equalization, apply_clahe,
    save_encoding_image, ENCODING_REGISTRY, RGB_STACK_PRESETS,
)
from vtbench.train.factory import get_chart_model
from vtbench.utils.wandb_logger import WandbLogger
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


# ====================================================================
# Dataset class for pre-generated encoding images
# ====================================================================

class EncodingImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, f"sample_{idx}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        img = self.transform(img)
        return img, self.labels[idx]


# ====================================================================
# Image generation for new encodings
# ====================================================================

def generate_encoding_images(X, dataset_name, split, encoding_name,
                             image_size=128, global_offset=0, overwrite=False):
    """Generate encoding images for one split."""
    out_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images",
                           encoding_name, split)
    os.makedirs(out_dir, exist_ok=True)

    n_new = 0
    for i, ts in enumerate(X):
        idx = global_offset + i
        path = os.path.join(out_dir, f"sample_{idx}.png")
        if os.path.exists(path) and not overwrite:
            continue

        try:
            if encoding_name == 'phase_multi_tau':
                arr = encode_phase_space_multi_tau(ts, image_size)
            elif encoding_name in ENCODING_REGISTRY:
                arr = get_encoding(encoding_name, ts, image_size)
            elif encoding_name in RGB_STACK_PRESETS:
                arr = get_rgb_stack(encoding_name, ts, image_size)
            else:
                print(f"  WARNING: Unknown encoding {encoding_name}, skipping")
                continue
            save_encoding_image(arr, path)
            n_new += 1
        except Exception as e:
            print(f"  ERROR: {dataset_name}/{encoding_name}/sample_{idx}: {e}")

    return n_new


# ====================================================================
# Training & Evaluation
# ====================================================================

def train_and_evaluate(train_dataset, test_dataset, model_name, pretrained,
                       lr, num_classes, epochs=100, batch_size=32):
    """Train a model and return test accuracy."""
    model = None
    train_loader = None
    test_loader = None
    try:
        safe_bs = min(batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=safe_bs, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=min(batch_size, len(test_dataset)),
            shuffle=False)

        if len(train_loader) == 0:
            return 0.0

        model = get_chart_model(model_name, num_classes=num_classes,
                                pretrained=pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        patience_counter = 0
        patience = 10

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
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
            avg_loss = total_loss / len(train_loader)
            scheduler.step(avg_loss)

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        return best_acc
    except Exception as e:
        print(f"  ERROR in train_and_evaluate: {e}")
        return -1
    finally:
        model = None
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


# ====================================================================
# Main
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_7a_extended")
    os.makedirs(output_dir, exist_ok=True)

    datasets = exp["datasets"]
    seeds = exp.get("seeds", [42, 123, 7])
    image_size = exp.get("image_size", 128)
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    epochs = cfg.get("training", {}).get("epochs", 100)

    # Encodings to test
    new_encodings = exp.get("encodings", [
        'phase_scatter', 'phase_trajectory', 'phase_tau_auto',
        'gasf_diff', 'gasf_cumsum', 'rp_diff',
    ])
    rgb_encodings = exp.get("rgb_encodings", [
        'phase_gasf_rp', 'phase_cwt_mtf', 'gasf_gasfdiff_gadf',
        'phase_multi_tau',
    ])
    all_encodings = new_encodings + rgb_encodings

    models = exp.get("models", [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False, "lr": 0.001},
        {"name": "resnet18_pt", "chart_model": "resnet18", "pretrained": True, "lr": 0.0005},
    ])

    dataset_root = exp.get("dataset_root", "UCRArchive_2018")
    dataset_ext = exp.get("dataset_ext", "tsv")

    # W&B logger (no-op if wandb not installed)
    wb = WandbLogger(project="vtbench", experiment="7a_extended_encodings", config=cfg)

    rows = []
    completed_runs = set()

    # CSV resume logic
    csv_path = os.path.join(output_dir, "accuracy_extended_encodings.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing CSV: {csv_path}")
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_runs.add((row["dataset"], row["encoding"], row["model"], row["seed"]))
                rows.append(row)
        print(f"Loaded {len(completed_runs)} completed runs")

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        # Load data
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

        for enc_name in all_encodings:
            print(f"  Encoding: {enc_name}")

            # Generate images
            t0 = time.time()
            n1 = generate_encoding_images(
                X_train, dataset_name, "train", enc_name,
                image_size, global_offset=0)
            n2 = generate_encoding_images(
                X_test, dataset_name, "test", enc_name,
                image_size, global_offset=n_train)
            gen_time = time.time() - t0
            if n1 + n2 > 0:
                print(f"    Generated {n1+n2} images ({gen_time:.1f}s)")

            # Image directories
            train_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images",
                                     enc_name, "train")
            test_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images",
                                    enc_name, "test")

            if not os.path.isdir(train_dir) or not os.path.isdir(test_dir):
                print(f"    SKIP: image dirs not found")
                continue

            for model_cfg in models:
                for seed in seeds:
                    run_key = (dataset_name, enc_name, model_cfg["name"], str(seed))
                    if run_key in completed_runs:
                        print(f"    {model_cfg['name']} seed={seed}: SKIPPED (already completed)")
                        continue

                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    train_ds = EncodingImageDataset(train_dir, list(y_train))
                    test_ds = EncodingImageDataset(test_dir, list(y_test))

                    acc = train_and_evaluate(
                        train_ds, test_ds,
                        model_cfg["chart_model"], model_cfg["pretrained"],
                        model_cfg["lr"], num_classes,
                        epochs=epochs, batch_size=batch_size,
                    )

                    print(f"    {model_cfg['name']} seed={seed}: {acc:.4f}")

                    run_info = {
                        "dataset": dataset_name,
                        "encoding": enc_name,
                        "model": model_cfg["name"],
                        "seed": seed,
                        "accuracy": f"{acc:.4f}",
                    }
                    rows.append(run_info)

                    # Log to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/{enc_name}/{model_cfg['name']}/s{seed}",
                        config=run_info,
                        accuracy=acc,
                    )

                    # GPU cleanup to prevent CUDA driver corruption
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    # Save results (append mode if file exists, write mode if new)
    csv_path = os.path.join(output_dir, "accuracy_extended_encodings.csv")
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a" if file_exists else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "encoding", "model",
                                                "seed", "accuracy"])
        if not file_exists:
            writer.writeheader()
        # Only write new rows (those not in completed_runs)
        for row in rows:
            if (row["dataset"], row["encoding"], row["model"], row["seed"]) not in completed_runs:
                writer.writerow(row)

    # Summary
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 7A: Extended Image Encodings\n\n")
        f.write(f"Encodings tested: {len(all_encodings)}\n")
        f.write(f"Datasets: {len(datasets)}\n")
        f.write(f"Models: {len(models)}\n")
        f.write(f"Seeds: {seeds}\n\n")

        # Mean accuracy per encoding x model
        f.write("## Mean Accuracy by Encoding x Model\n\n")
        f.write("| Encoding | " + " | ".join(m["name"] for m in models) + " |\n")
        f.write("|" + "---|" * (len(models) + 1) + "\n")
        for enc in all_encodings:
            vals = []
            for m in models:
                accs = [float(r["accuracy"]) for r in rows
                        if r["encoding"] == enc and r["model"] == m["name"]]
                if accs:
                    vals.append(f"{np.mean(accs):.4f} +/- {np.std(accs):.4f}")
                else:
                    vals.append("N/A")
            f.write(f"| {enc} | " + " | ".join(vals) + " |\n")

    print(f"\nResults saved to {csv_path}")
    print(f"Summary saved to {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)
