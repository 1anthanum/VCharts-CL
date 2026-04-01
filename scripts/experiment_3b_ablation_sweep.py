import argparse
import copy
import csv
import os
import random

import numpy as np
import torch
import yaml
from torchvision import transforms
import matplotlib.pyplot as plt

from vtbench.train.trainer import train_model
from vtbench.train.evaluate import evaluate_model
from vtbench.data.loader import create_dataloaders
from vtbench.utils.ablation import apply_ablation, ablation_type_name, chart_dir_for


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dataset_entries(exp_cfg):
    exp = exp_cfg["experiment"]
    dataset_root = exp.get("dataset_root", ".")
    dataset_ext = exp.get("dataset_ext", "tsv")
    entries = []
    for item in exp.get("datasets", []):
        if isinstance(item, str):
            name = item
            train_path = os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}")
            test_path = os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}")
            entries.append(
                {
                    "name": name,
                    "train_path": train_path,
                    "test_path": test_path,
                    "format": exp.get("dataset_format", "ucr"),
                }
            )
        else:
            name = item["name"]
            train_path = item.get(
                "train_path",
                os.path.join(dataset_root, name, f"{name}_TRAIN.{dataset_ext}"),
            )
            test_path = item.get(
                "test_path",
                os.path.join(dataset_root, name, f"{name}_TEST.{dataset_ext}"),
            )
            entries.append(
                {
                    "name": name,
                    "train_path": train_path,
                    "test_path": test_path,
                    "format": item.get("format", exp.get("dataset_format", "ucr")),
                }
            )
    return entries


def _build_run_config(exp_cfg, dataset_entry, chart_type):
    chart_defaults = exp_cfg.get("chart", {})
    bar_mode = chart_defaults.get("bar_mode", "border")
    scatter_mode = chart_defaults.get("scatter_mode", "plain")

    config = {
        "dataset": {
            "name": dataset_entry["name"],
            "train_path": dataset_entry["train_path"],
            "test_path": dataset_entry["test_path"],
            "format": dataset_entry.get("format", "ucr"),
        },
        "image_generation": copy.deepcopy(exp_cfg.get("image_generation", {})),
        "model": copy.deepcopy(exp_cfg.get("model", {})),
        "training": copy.deepcopy(exp_cfg.get("training", {})),
        "chart_branches": {
            "branch_1": {
                "chart_type": chart_type,
                "color_mode": chart_defaults.get("color_mode", "color"),
                "label_mode": chart_defaults.get("label_mode", "with_label"),
            }
        },
    }

    if chart_type == "bar":
        config["chart_branches"]["branch_1"]["bar_mode"] = bar_mode
    if chart_type == "scatter":
        config["chart_branches"]["branch_1"]["scatter_mode"] = scatter_mode

    return config


def _base_transforms(config):
    dataset_format = config["dataset"].get("format", "ucr")
    preserve_mv = bool(config.get("image_generation", {}).get("preserve_multivariate_size", True))
    chart_model = str(config["model"].get("chart_model", "")).lower()
    is_resnet18 = chart_model == "resnet18"
    pretrained = is_resnet18 and config["model"].get("pretrained", False)

    if dataset_format == "uea" and preserve_mv:
        base = [transforms.ToTensor()]
    else:
        base = [transforms.Resize((128, 128)), transforms.ToTensor()]

    if pretrained:
        base.append(
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
        )

    return base


def _ablation_transform(chart_type, ablations_cfg):
    return lambda img: apply_ablation(img, chart_type, ablations_cfg)


def _set_chart_transform(chart_loaders, transform):
    if isinstance(chart_loaders, list):
        for loader in chart_loaders:
            loader.dataset.transform = transform
    else:
        chart_loaders.dataset.transform = transform


def _set_chart_dirs(chart_loaders, chart_branches, dataset_name, ablated_root, split):
    branch_cfgs = list(chart_branches.values())
    if isinstance(chart_loaders, list):
        if len(branch_cfgs) != len(chart_loaders):
            raise ValueError("Chart branch count does not match chart loaders.")
        for loader, cfg in zip(chart_loaders, branch_cfgs):
            chart_dir = chart_dir_for(
                ablated_root,
                dataset_name,
                cfg["chart_type"],
                cfg.get("color_mode", "color"),
                cfg.get("label_mode", "with_label"),
                split,
                bar_mode=cfg.get("bar_mode", "border"),
                scatter_mode=cfg.get("scatter_mode", "plain"),
            )
            if not os.path.isdir(chart_dir):
                raise FileNotFoundError(f"Ablated chart dir not found: {chart_dir}")
            loader.dataset.chart_dir = chart_dir
    else:
        cfg = branch_cfgs[0]
        chart_dir = chart_dir_for(
            ablated_root,
            dataset_name,
            cfg["chart_type"],
            cfg.get("color_mode", "color"),
            cfg.get("label_mode", "with_label"),
            split,
            bar_mode=cfg.get("bar_mode", "border"),
            scatter_mode=cfg.get("scatter_mode", "plain"),
        )
        if not os.path.isdir(chart_dir):
            raise FileNotFoundError(f"Ablated chart dir not found: {chart_dir}")
        chart_loaders.dataset.chart_dir = chart_dir


def _evaluate_with_transform(model, config, transform=None, ablated_root=None):
    loaders = create_dataloaders(config)
    if ablated_root:
        _set_chart_dirs(
            loaders["test"]["chart"],
            config["chart_branches"],
            config["dataset"]["name"],
            ablated_root,
            "test",
        )
    if transform is not None:
        _set_chart_transform(loaders["test"]["chart"], transform)

    model_type = config["model"]["type"]
    if model_type == "single_modal_chart":
        eval_type = "single_chart"
    elif model_type == "two_branch":
        eval_type = "two_branch"
    elif model_type == "multi_modal_chart":
        eval_type = "multi_chart"
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return evaluate_model(
        model,
        loaders["test"]["chart"],
        loaders["test"]["numerical"],
        eval_type,
    )


def _sweep_spec(chart_type, sweeps_cfg):
    if chart_type in ("line", "area"):
        group = "line_area"
        param = "blur_radius"
    elif chart_type == "bar":
        group = "bar"
        param = "merge_scale"
    elif chart_type == "scatter":
        group = "scatter"
        param = "fade_alpha"
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    values = sweeps_cfg.get(group, {}).get(param)
    if not values:
        raise ValueError(f"Missing sweep values for {group}.{param}")
    return group, param, values


def _plot_sweep_group(dataset_name, entries, group_types, title_suffix, output_path):
    ordered = [e for e in entries if e["chart_type"] in group_types]
    if not ordered:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for chart_type in group_types:
        subset = [e for e in ordered if e["chart_type"] == chart_type]
        if not subset:
            continue
        subset = sorted(subset, key=lambda e: e["sweep_value"])
        x_vals = [e["sweep_value"] for e in subset]
        y_vals = [e["delta_accuracy"] for e in subset]
        ax.plot(x_vals, y_vals, marker="o", label=chart_type)

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("Delta accuracy")
    ax.set_xlabel("Ablation strength")
    ax.set_title(f"{dataset_name} - {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _set_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _aggregate_results(rows):
    grouped = {}
    for row in rows:
        key = (
            row["dataset"],
            row["chart_type"],
            row["ablation_type"],
            row["sweep_param"],
            row["sweep_value"],
        )
        grouped.setdefault(
            key,
            {"baseline": [], "ablated": [], "delta": []},
        )
        grouped[key]["baseline"].append(row["baseline_accuracy"])
        grouped[key]["ablated"].append(row["ablated_accuracy"])
        grouped[key]["delta"].append(row["delta_accuracy"])

    summary = []
    for key, values in grouped.items():
        baseline = np.array(values["baseline"], dtype=np.float32)
        ablated = np.array(values["ablated"], dtype=np.float32)
        delta = np.array(values["delta"], dtype=np.float32)
        summary.append(
            {
                "dataset": key[0],
                "chart_type": key[1],
                "ablation_type": key[2],
                "sweep_param": key[3],
                "sweep_value": key[4],
                "baseline_mean": float(np.mean(baseline)),
                "baseline_std": float(np.std(baseline, ddof=1)) if len(baseline) > 1 else 0.0,
                "ablated_mean": float(np.mean(ablated)),
                "ablated_std": float(np.std(ablated, ddof=1)) if len(ablated) > 1 else 0.0,
                "delta_mean": float(np.mean(delta)),
                "delta_std": float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0,
                "runs": len(delta),
            }
        )
    return summary


def _plot_sweep_group_mean(dataset_name, entries, group_types, title_suffix, output_path):
    ordered = [e for e in entries if e["chart_type"] in group_types]
    if not ordered:
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for chart_type in group_types:
        subset = [e for e in ordered if e["chart_type"] == chart_type]
        if not subset:
            continue
        subset = sorted(subset, key=lambda e: e["sweep_value"])
        x_vals = [e["sweep_value"] for e in subset]
        y_vals = [e["delta_mean"] for e in subset]
        y_err = [e["delta_std"] for e in subset]
        ax.errorbar(x_vals, y_vals, yerr=y_err, marker="o", label=chart_type, capsize=2)

    ax.axhline(0.0, color="#333333", linewidth=1.0)
    ax.set_ylabel("Delta accuracy (mean)")
    ax.set_xlabel("Ablation strength")
    ax.set_title(f"{dataset_name} - {title_suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_3b_sweep")
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    sweeps_cfg = cfg.get("sweeps", {})
    base_ablations_cfg = cfg.get("ablations", {})
    runs = int(exp.get("runs", 1))
    seed = exp.get("seed")

    results = []

    for run_id in range(1, runs + 1):
        run_seed = None if seed is None else int(seed) + run_id - 1
        _set_seeds(run_seed)
        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]
            for chart_type in chart_types:
                run_config = _build_run_config(cfg, dataset_entry, chart_type)
                train_config = copy.deepcopy(run_config)
                train_config["image_generation"].setdefault("generate_images", True)
                train_config["image_generation"].setdefault("overwrite_existing", False)

                model = train_model(train_config)

                eval_config = copy.deepcopy(run_config)
                eval_config["image_generation"]["generate_images"] = False
                eval_config["image_generation"]["overwrite_existing"] = False

                baseline_transform = transforms.Compose(_base_transforms(eval_config))
                baseline = _evaluate_with_transform(model, eval_config, transform=baseline_transform)
                baseline_acc = float(baseline["accuracy"])

                sweep_group, sweep_param, sweep_values = _sweep_spec(chart_type, sweeps_cfg)

                for value in sweep_values:
                    ablations_cfg = copy.deepcopy(base_ablations_cfg)
                    ablations_cfg.setdefault(sweep_group, {})
                    ablations_cfg[sweep_group][sweep_param] = value

                    ablation_fn = _ablation_transform(chart_type, ablations_cfg)
                    transform = transforms.Compose([ablation_fn] + _base_transforms(eval_config))
                    ablated = _evaluate_with_transform(model, eval_config, transform=transform)

                    ablated_acc = float(ablated["accuracy"])
                    delta = ablated_acc - baseline_acc

                    results.append(
                        {
                            "run_id": run_id,
                            "seed": run_seed,
                            "dataset": dataset_name,
                            "chart_type": chart_type,
                            "ablation_type": ablation_type_name(chart_type),
                            "sweep_param": sweep_param,
                            "sweep_value": value,
                            "baseline_accuracy": baseline_acc,
                            "ablated_accuracy": ablated_acc,
                            "delta_accuracy": delta,
                        }
                    )

    csv_path = os.path.join(output_dir, "ablation_sweep_results_runs.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "seed",
                "dataset",
                "chart_type",
                "ablation_type",
                "sweep_param",
                "sweep_value",
                "baseline_accuracy",
                "ablated_accuracy",
                "delta_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary = _aggregate_results(results)

    summary_csv = os.path.join(output_dir, "ablation_sweep_results_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "chart_type",
                "ablation_type",
                "sweep_param",
                "sweep_value",
                "baseline_mean",
                "baseline_std",
                "ablated_mean",
                "ablated_std",
                "delta_mean",
                "delta_std",
                "runs",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    md_path = os.path.join(output_dir, "ablation_sweep_results_summary.md")
    with open(md_path, "w") as f:
        f.write("| dataset | encoding | ablation_type | param | value | baseline_mean | ablated_mean | delta_mean | runs |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in summary:
            f.write(
                f"| {row['dataset']} | {row['chart_type']} | {row['ablation_type']} | "
                f"{row['sweep_param']} | {row['sweep_value']} | {row['baseline_mean']:.4f} | "
                f"{row['ablated_mean']:.4f} | {row['delta_mean']:.4f} | {row['runs']} |\n"
            )

    for dataset_entry in datasets:
        name = dataset_entry["name"]
        ds_entries = [r for r in summary if r["dataset"] == name]
        _plot_sweep_group_mean(
            name,
            ds_entries,
            ["line", "area"],
            "Line/Area Sweep (mean ± std)",
            os.path.join(plots_dir, f"{name}_line_area_sweep.png"),
        )
        _plot_sweep_group_mean(
            name,
            ds_entries,
            ["bar", "scatter"],
            "Bar/Scatter Sweep (mean ± std)",
            os.path.join(plots_dir, f"{name}_bar_scatter_sweep.png"),
        )

    print(f"Saved sweep results to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3B: Ablation Strength Sweep")
    parser.add_argument("--config", required=True, help="Path to sweep config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
