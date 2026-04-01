#!/usr/bin/env python
"""
Experiment 5L: Ensemble Voting
===============================
Trains independent models (per chart_type x backbone), then combines
predictions via hard majority voting and soft probability averaging.

Ensemble strategies:
  1. chart_ensemble: fix model, vote across chart types
  2. model_ensemble: fix chart type, vote across models
  3. full_ensemble: vote across all (chart_type x model) combos

Usage
-----
    python scripts/experiment_5l_ensemble.py \
        --config vtbench/config/experiment_5l_ensemble.yaml
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
import torch.nn.functional as F
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
    _set_seeds,
)


def _get_predictions(model, loader, device):
    """Get predicted labels and class probabilities for entire test set."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.vstack(all_probs), np.array(all_labels)


def _hard_vote(pred_list):
    """Majority voting across multiple prediction arrays."""
    stacked = np.stack(pred_list, axis=0)  # (num_models, num_samples)
    from scipy import stats
    voted, _ = stats.mode(stacked, axis=0, keepdims=False)
    return voted.flatten()


def _soft_vote(prob_list):
    """Average probabilities and take argmax."""
    avg_probs = np.mean(prob_list, axis=0)  # (num_samples, num_classes)
    return avg_probs.argmax(axis=1)


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_5l_ensemble")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "scatter"])
    datasets = _dataset_entries(cfg)
    seeds = exp.get("seeds", [42])
    model_configs = exp.get("model_configs", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    accuracy_rows = []

    # Initialize W&B logger
    wb = WandbLogger(project="vtbench", experiment="5l_ensemble", config=cfg)
    hb = Heartbeat("5l")

    total_runs = len(seeds) * len(datasets) * len(model_configs) * len(chart_types)
    run_idx = 0

    for seed in seeds:
        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]
            print(f"\n{'=' * 60}")
            print(f"Dataset: {dataset_name} (seed={seed})")
            print(f"{'=' * 60}")

            # Train all individual models and collect predictions
            individual_results = {}  # (model_name, chart_type) -> (preds, probs, labels)

            for mc in model_configs:
                model_name = mc["name"]
                for chart_type in chart_types:
                    run_idx += 1
                    print(f"\n  Training: {model_name} / {chart_type}")

                    hb.pulse(
                        dataset=dataset_name,
                        encoding=chart_type,
                        model=model_name,
                        run=f"{run_idx}/{total_runs}",
                    )

                    run_config = _build_run_config(cfg, dataset_entry, chart_type)
                    run_config["model"]["chart_model"] = mc["chart_model"]
                    run_config["model"]["pretrained"] = mc.get("pretrained", False)
                    run_config["training"]["learning_rate"] = mc.get("lr", 0.001)
                    _ensure_base_images(run_config)

                    _set_seeds(seed)
                    try:
                        model = train_model(run_config).to(device)
                        model.eval()
                        loaders = create_dataloaders(run_config)
                        test_loader = loaders["test"]["chart"]
                        preds, probs, labels = _get_predictions(model, test_loader, device)

                        acc = np.mean(preds == labels)
                        print(f"    Individual accuracy: {acc:.4f}")

                        individual_results[(model_name, chart_type)] = (preds, probs, labels)

                        accuracy_rows.append({
                            "seed": seed, "dataset": dataset_name,
                            "ensemble_type": "individual",
                            "members": f"{model_name}/{chart_type}",
                            "voting": "none", "accuracy": acc,
                        })
                        # Log clean accuracy to W&B
                        wb.log_run_result(
                            name=f"{dataset_name}/individual/{model_name}/{chart_type}/s{seed}",
                            config={"ensemble_type": "individual",
                                    "model": model_name, "chart_type": chart_type,
                                    "seed": seed, "dataset": dataset_name},
                            accuracy=acc,
                        )

                        del model, loaders, test_loader
                    except Exception as e:
                        print(f"    FAILED: {e}")
                        continue

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            if not individual_results:
                continue

            # Get ground truth labels
            labels = list(individual_results.values())[0][2]

            # --- Ensemble 1: Chart-type ensemble (per model) ---
            for mc in model_configs:
                model_name = mc["name"]
                pred_list = []
                prob_list = []
                members = []
                for ct in chart_types:
                    if (model_name, ct) in individual_results:
                        p, pr, _ = individual_results[(model_name, ct)]
                        pred_list.append(p)
                        prob_list.append(pr)
                        members.append(ct)

                if len(pred_list) >= 2:
                    hard_preds = _hard_vote(pred_list)
                    soft_preds = _soft_vote(prob_list)
                    hard_acc = np.mean(hard_preds == labels)
                    soft_acc = np.mean(soft_preds == labels)
                    member_str = f"{model_name}/[{'+'.join(members)}]"
                    print(f"\n  Chart ensemble ({member_str}):")
                    print(f"    Hard vote: {hard_acc:.4f}  Soft vote: {soft_acc:.4f}")

                    accuracy_rows.append({
                        "seed": seed, "dataset": dataset_name,
                        "ensemble_type": "chart_ensemble",
                        "members": member_str,
                        "voting": "hard", "accuracy": hard_acc,
                    })
                    # Log chart_ensemble hard vote to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/chart_ensemble/{model_name}/hard/s{seed}",
                        config={"ensemble_type": "chart_ensemble", "voting": "hard",
                                "model": model_name, "seed": seed,
                                "dataset": dataset_name},
                        accuracy=hard_acc,
                    )
                    accuracy_rows.append({
                        "seed": seed, "dataset": dataset_name,
                        "ensemble_type": "chart_ensemble",
                        "members": member_str,
                        "voting": "soft", "accuracy": soft_acc,
                    })
                    # Log chart_ensemble soft vote to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/chart_ensemble/{model_name}/soft/s{seed}",
                        config={"ensemble_type": "chart_ensemble", "voting": "soft",
                                "model": model_name, "seed": seed,
                                "dataset": dataset_name},
                        accuracy=soft_acc,
                    )

            # --- Ensemble 2: Model ensemble (per chart type) ---
            for ct in chart_types:
                pred_list = []
                prob_list = []
                members = []
                for mc in model_configs:
                    if (mc["name"], ct) in individual_results:
                        p, pr, _ = individual_results[(mc["name"], ct)]
                        pred_list.append(p)
                        prob_list.append(pr)
                        members.append(mc["name"])

                if len(pred_list) >= 2:
                    hard_preds = _hard_vote(pred_list)
                    soft_preds = _soft_vote(prob_list)
                    hard_acc = np.mean(hard_preds == labels)
                    soft_acc = np.mean(soft_preds == labels)
                    member_str = f"[{'+'.join(members)}]/{ct}"
                    print(f"\n  Model ensemble ({member_str}):")
                    print(f"    Hard vote: {hard_acc:.4f}  Soft vote: {soft_acc:.4f}")

                    accuracy_rows.append({
                        "seed": seed, "dataset": dataset_name,
                        "ensemble_type": "model_ensemble",
                        "members": member_str,
                        "voting": "hard", "accuracy": hard_acc,
                    })
                    # Log model_ensemble hard vote to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/model_ensemble/{ct}/hard/s{seed}",
                        config={"ensemble_type": "model_ensemble", "voting": "hard",
                                "chart_type": ct, "seed": seed,
                                "dataset": dataset_name},
                        accuracy=hard_acc,
                    )
                    accuracy_rows.append({
                        "seed": seed, "dataset": dataset_name,
                        "ensemble_type": "model_ensemble",
                        "members": member_str,
                        "voting": "soft", "accuracy": soft_acc,
                    })
                    # Log model_ensemble soft vote to W&B
                    wb.log_run_result(
                        name=f"{dataset_name}/model_ensemble/{ct}/soft/s{seed}",
                        config={"ensemble_type": "model_ensemble", "voting": "soft",
                                "chart_type": ct, "seed": seed,
                                "dataset": dataset_name},
                        accuracy=soft_acc,
                    )

            # --- Ensemble 3: Full ensemble ---
            all_preds = [r[0] for r in individual_results.values()]
            all_probs = [r[1] for r in individual_results.values()]
            if len(all_preds) >= 3:
                hard_preds = _hard_vote(all_preds)
                soft_preds = _soft_vote(all_probs)
                hard_acc = np.mean(hard_preds == labels)
                soft_acc = np.mean(soft_preds == labels)
                print(f"\n  Full ensemble ({len(all_preds)} models):")
                print(f"    Hard vote: {hard_acc:.4f}  Soft vote: {soft_acc:.4f}")

                accuracy_rows.append({
                    "seed": seed, "dataset": dataset_name,
                    "ensemble_type": "full_ensemble",
                    "members": f"all_{len(all_preds)}",
                    "voting": "hard", "accuracy": hard_acc,
                })
                # Log full_ensemble hard vote to W&B
                wb.log_run_result(
                    name=f"{dataset_name}/full_ensemble/all_{len(all_preds)}/hard/s{seed}",
                    config={"ensemble_type": "full_ensemble", "voting": "hard",
                            "num_members": len(all_preds), "seed": seed,
                            "dataset": dataset_name},
                    accuracy=hard_acc,
                )
                accuracy_rows.append({
                    "seed": seed, "dataset": dataset_name,
                    "ensemble_type": "full_ensemble",
                    "members": f"all_{len(all_preds)}",
                    "voting": "soft", "accuracy": soft_acc,
                })
                # Log full_ensemble soft vote to W&B
                wb.log_run_result(
                    name=f"{dataset_name}/full_ensemble/all_{len(all_preds)}/soft/s{seed}",
                    config={"ensemble_type": "full_ensemble", "voting": "soft",
                            "num_members": len(all_preds), "seed": seed,
                            "dataset": dataset_name},
                    accuracy=soft_acc,
                )

    # Save CSV
    csv_path = os.path.join(output_dir, "accuracy_results.csv")
    fieldnames = ["seed", "dataset", "ensemble_type", "members", "voting", "accuracy"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(accuracy_rows)
    print(f"\nResults saved to {csv_path}")

    hb.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 5L: Ensemble Voting")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
