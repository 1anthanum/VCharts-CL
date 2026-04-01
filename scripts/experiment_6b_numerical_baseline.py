#!/usr/bin/env python
"""
Experiment 6B: Pure Numerical Baselines
=========================================
Run FCN, Transformer, and OS-CNN directly on raw time series (no images).
This provides the critical baseline for answering: "Does converting TS
to images actually help, compared to processing numbers directly?"

Usage:
    python scripts/experiment_6b_numerical_baseline.py --config vtbench/config/experiment_6b_numerical.yaml
"""

import argparse
import csv
import os
import sys
import time
import yaml
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vtbench.data.loader import read_ucr
from vtbench.data.chart_generator import NumericalDataset
from vtbench.models.numerical.fcn import NumericalFCN
from vtbench.models.numerical.transformer import NumericalTransformer
from vtbench.models.numerical.oscnn import NumericalOSCNN
from vtbench.utils.wandb_logger import WandbLogger

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_numerical_model(name, input_dim, num_classes):
    """Create a numerical model by name."""
    if name == 'fcn':
        return NumericalFCN(input_dim=input_dim, output_dim=num_classes)
    elif name == 'transformer':
        return NumericalTransformer(input_dim=input_dim, output_dim=num_classes)
    elif name == 'oscnn':
        # OSCNN expects input_channels (default=1 for univariate TS),
        # not input_dim (time series length)
        return NumericalOSCNN(input_channels=1, output_dim=num_classes)
    else:
        raise ValueError(f"Unknown numerical model: {name}")


def train_and_evaluate(model, train_loader, test_loader, epochs, lr, device):
    """Train and return test accuracy."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        correct = 0
        total = 0

        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total if total > 0 else 0

        # Evaluate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_loss)

        print(f"  [Epoch {epoch+1}] Train: {train_acc:.2%}, Test: {val_acc:.2%}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"  Early stopping.")
                break

    return best_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exp = cfg['experiment']
    output_dir = exp['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="6b_numerical_baseline", config=cfg)

    datasets = exp['datasets']
    seeds = exp.get('seeds', [42, 123, 7])
    models = exp.get('models', ['fcn', 'transformer', 'oscnn'])
    epochs = cfg.get('training', {}).get('epochs', 100)
    batch_size = cfg.get('training', {}).get('batch_size', 64)
    lr = cfg.get('training', {}).get('learning_rate', 0.001)

    csv_path = os.path.join(output_dir, "numerical_baselines.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["seed", "dataset", "model", "accuracy",
                      "train_samples", "test_samples", "num_classes", "ts_length"])

    total_runs = len(datasets) * len(seeds) * len(models)
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
            print(f"SKIP {ds_name}: not found")
            continue

        n_train, n_test = len(X_train), len(X_test)
        num_classes = len(set(y_train))
        ts_length = X_train.shape[-1] if hasattr(X_train, 'shape') else len(X_train[0])

        print(f"\nDataset: {ds_name} (train={n_train}, test={n_test}, "
              f"classes={num_classes}, length={ts_length})")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            train_ds = NumericalDataset(X_train, y_train)
            test_ds = NumericalDataset(X_test, y_test)
            safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
            train_loader = DataLoader(train_ds, batch_size=safe_bs,
                                       shuffle=True, drop_last=True)
            test_loader = DataLoader(test_ds, batch_size=min(batch_size, len(test_ds)),
                                       shuffle=False)

            for model_name in models:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {ds_name} | {model_name} | seed={seed}")

                try:
                    model = get_numerical_model(model_name, ts_length, num_classes)
                    accuracy = train_and_evaluate(
                        model, train_loader, test_loader, epochs, lr, device
                    )
                    print(f"  -> Accuracy: {accuracy:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    accuracy = -1

                writer.writerow([
                    seed, ds_name, model_name, f"{accuracy:.6f}",
                    n_train, n_test, num_classes, ts_length
                ])
                csv_file.flush()

                # Log to W&B
                run_info = {
                    "dataset": ds_name,
                    "model": model_name,
                    "seed": seed,
                }
                wb.log_run_result(
                    name=f"{ds_name}_{model_name}_seed{seed}",
                    config=run_info,
                    accuracy=accuracy,
                )

    csv_file.close()
    print(f"\nResults saved to: {csv_path}")

    # Finish W&B logging
    wb.finish()


if __name__ == "__main__":
    main()
