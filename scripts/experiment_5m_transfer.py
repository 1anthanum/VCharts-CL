#!/usr/bin/env python
"""
Experiment 5M: Cross-Dataset Transfer Learning
================================================
Pre-trains on large chart-image datasets, then fine-tunes on small
target datasets. Tests whether chart visual features transfer across
different time-series domains.

Usage
-----
    python scripts/experiment_5m_transfer.py \
        --config vtbench/config/experiment_5m_transfer.yaml
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

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from vtbench.train.trainer import train_model
from vtbench.train.factory import get_chart_model
from vtbench.data.loader import create_dataloaders
from vtbench.utils.wandb_logger import WandbLogger
from vtbench.utils.heartbeat import Heartbeat
from vtbench.utils.experiment_helpers import (
    _dataset_entries,
    _build_run_config,
    _ensure_base_images,
    _evaluate_accuracy,
    _set_seeds,
)


def _pretrain_on_source(cfg, source_entry, chart_type, seed, device):
    """Pre-train a model on a source dataset and return the feature extractor."""
    run_config = _build_run_config(cfg, source_entry, chart_type)
    _ensure_base_images(run_config)
    _set_seeds(seed)
    model = train_model(run_config).to(device)
    return model


def _finetune_on_target(pretrained_model, cfg, target_entry, chart_type,
                        strategy, seed, device):
    """Fine-tune a pretrained model on a target dataset."""
    run_config = _build_run_config(cfg, target_entry, chart_type)
    _ensure_base_images(run_config)

    loaders = create_dataloaders(run_config)
    train_loader = loaders['train']['chart']
    val_loader = loaders['val']['chart']
    test_loader = loaders['test']['chart']

    labels = [label for _, label in train_loader.dataset]
    num_classes = len(set(labels))

    # Clone the pretrained model's feature layers
    model = copy.deepcopy(pretrained_model)

    # Probe actual feature dimension with a dummy forward pass on target data
    # (DeepCNN uses LazyLinear — feature dim depends on input image size)
    model = model.to(device)
    model.eval()
    try:
        sample_img, _ = train_loader.dataset[0]
        with torch.no_grad():
            # Temporarily set classifier to Identity to get feature dim
            old_classifier = model.classifier
            model.classifier = nn.Identity()
            dummy_out = model(sample_img.unsqueeze(0).to(device))
            feature_dim = dummy_out.shape[-1]
            model.classifier = old_classifier  # restore
    except Exception:
        # Fallback: use model's feature_dim attribute
        feature_dim = getattr(model, 'feature_dim', 256)

    # Replace classifier for new number of classes using probed feature_dim
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(feature_dim, num_classes),
    ).to(device)

    model = model.to(device)

    # Freeze backbone if requested
    freeze = strategy.get("freeze_backbone", False)
    if freeze:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False

    lr = strategy.get("lr", 0.001)
    epochs = strategy.get("epochs", 50)

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_train, lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    patience = 10
    trigger_times = 0
    best_val_acc = 0

    _set_seeds(seed)

    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, lbls in train_loader:
            images, lbls = images.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += lbls.size(0)
            correct += predicted.eq(lbls).sum().item()

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.amp.autocast("cuda", enabled=use_amp):
            with torch.no_grad():
                for images, lbls in val_loader:
                    images, lbls = images.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                    outputs = model(images)
                    loss = criterion(outputs, lbls)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += lbls.size(0)
                    val_correct += predicted.eq(lbls).sum().item()

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        val_loss_avg = val_loss / max(len(val_loader), 1)

        print(f"[Epoch {epoch+1}] Train Acc: {train_acc:.1f}%, Val Acc: {val_acc:.1f}%")

        scheduler.step(val_loss_avg)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model, test_loader


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5m_transfer")
    os.makedirs(output_dir, exist_ok=True)

    dataset_root = exp.get("dataset_root", "UCRArchive_2018")
    dataset_ext = exp.get("dataset_ext", "tsv")
    dataset_format = exp.get("dataset_format", "ucr")

    chart_types = exp.get("chart_types", ["line"])
    seeds = exp.get("seeds", [42])
    strategies = exp.get("finetune_strategies", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy_rows = []

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5m_transfer", config=cfg)
    hb = Heartbeat("5m")

    def _make_entry(name):
        return {
            "name": name,
            "train_path": os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}"),
            "test_path": os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}"),
            "format": dataset_format,
        }

    source_names = exp.get("source_datasets", [])
    target_names = exp.get("target_datasets", [])

    total_runs = len(seeds) * len(chart_types) * (
        len(target_names) + len(source_names) * len(target_names) * len(strategies)
    )
    run_idx = 0

    for seed in seeds:
        for chart_type in chart_types:
            # --- Baseline: train from scratch on target ---
            for target_name in target_names:
                run_idx += 1
                target_entry = _make_entry(target_name)
                print(f"\n{'=' * 60}")
                print(f"BASELINE: {target_name} from scratch (seed={seed}, {chart_type})")
                print(f"{'=' * 60}")

                hb.pulse(
                    dataset=target_name,
                    encoding="transfer",
                    model="scratch",
                    run=f"{run_idx}/{total_runs}",
                    source="scratch",
                )

                run_config = _build_run_config(cfg, target_entry, chart_type)
                _ensure_base_images(run_config)
                _set_seeds(seed)

                try:
                    model = train_model(run_config).to(device)
                    model.eval()
                    loaders = create_dataloaders(run_config)
                    test_loader = loaders["test"]["chart"]
                    acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
                    print(f"  Accuracy: {acc:.4f}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    acc = -1

                accuracy_rows.append({
                    "seed": seed, "target": target_name,
                    "source": "scratch", "strategy": "from_scratch",
                    "chart_type": chart_type, "accuracy": acc,
                })
                # Log baseline (from_scratch) accuracy to W&B
                wb.log_run_result(
                    name=f"{target_name}/scratch/from_scratch/{chart_type}/s{seed}",
                    config={"target": target_name, "source": "scratch",
                            "strategy": "from_scratch", "chart_type": chart_type,
                            "seed": seed},
                    accuracy=acc,
                )

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # --- Transfer: pre-train on source, fine-tune on target ---
            for source_name in source_names:
                source_entry = _make_entry(source_name)
                print(f"\n{'=' * 60}")
                print(f"PRE-TRAINING on {source_name} (seed={seed}, {chart_type})")
                print(f"{'=' * 60}")

                try:
                    pretrained = _pretrain_on_source(
                        cfg, source_entry, chart_type, seed, device
                    )
                except Exception as e:
                    print(f"  PRE-TRAINING FAILED: {e}")
                    continue

                for target_name in target_names:
                    target_entry = _make_entry(target_name)

                    for strategy in strategies:
                        run_idx += 1
                        strat_name = strategy["name"]
                        print(f"\n  FINE-TUNE: {source_name} -> {target_name} ({strat_name})")

                        hb.pulse(
                            dataset=target_name,
                            encoding="transfer",
                            model=strat_name,
                            run=f"{run_idx}/{total_runs}",
                            source=source_name,
                        )

                        try:
                            model, test_loader = _finetune_on_target(
                                pretrained, cfg, target_entry, chart_type,
                                strategy, seed, device,
                            )
                            model.eval()
                            acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
                            print(f"  Accuracy: {acc:.4f}")
                        except Exception as e:
                            print(f"  FINE-TUNE FAILED: {e}")
                            import traceback
                            traceback.print_exc()
                            acc = -1

                        accuracy_rows.append({
                            "seed": seed, "target": target_name,
                            "source": source_name, "strategy": strat_name,
                            "chart_type": chart_type, "accuracy": acc,
                        })
                        # Log transfer learning accuracy to W&B
                        wb.log_run_result(
                            name=f"{target_name}/{source_name}/{strat_name}/{chart_type}/s{seed}",
                            config={"target": target_name, "source": source_name,
                                    "strategy": strat_name, "chart_type": chart_type,
                                    "seed": seed},
                            accuracy=acc,
                        )

                        model = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                del pretrained
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    fieldnames = ["seed", "target", "source", "strategy", "chart_type", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nResults saved to {csv_path}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — Transfer Learning Accuracy")
    print(f"{'=' * 60}")
    from collections import defaultdict
    acc_map = defaultdict(list)
    for row in accuracy_rows:
        if row["accuracy"] >= 0:
            key = (row["target"], row["source"], row["strategy"])
            acc_map[key].append(row["accuracy"])

    prev_target = None
    for (tgt, src, strat), accs in sorted(acc_map.items()):
        if tgt != prev_target:
            print(f"\n  Target: {tgt}")
            prev_target = tgt
        print(f"    {src:>12s} / {strat:<18s}:  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    hb.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 5M: Transfer Learning")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
