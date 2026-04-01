import argparse
import copy
import os

import yaml
from PIL import Image

from vtbench.data.loader import create_dataloaders
from vtbench.utils.ablation import apply_ablation, chart_dir_for


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


def _ensure_base_images(config, generate_base_images):
    if not generate_base_images:
        return
    base_cfg = copy.deepcopy(config)
    base_cfg["image_generation"].setdefault("generate_images", True)
    base_cfg["image_generation"].setdefault("overwrite_existing", False)
    create_dataloaders(base_cfg)


def _ablated_dir_info(exp_cfg, dataset_name, chart_type, split):
    chart_defaults = exp_cfg.get("chart", {})
    return chart_dir_for(
        exp_cfg["experiment"].get("ablated_image_root", "chart_images_ablated"),
        dataset_name,
        chart_type,
        chart_defaults.get("color_mode", "color"),
        chart_defaults.get("label_mode", "with_label"),
        split,
        bar_mode=chart_defaults.get("bar_mode", "border"),
        scatter_mode=chart_defaults.get("scatter_mode", "plain"),
    )


def _base_dir_info(exp_cfg, dataset_name, chart_type, split):
    chart_defaults = exp_cfg.get("chart", {})
    return chart_dir_for(
        "chart_images",
        dataset_name,
        chart_type,
        chart_defaults.get("color_mode", "color"),
        chart_defaults.get("label_mode", "with_label"),
        split,
        bar_mode=chart_defaults.get("bar_mode", "border"),
        scatter_mode=chart_defaults.get("scatter_mode", "plain"),
    )


def generate_ablated_images(cfg):
    exp = cfg["experiment"]
    datasets = _dataset_entries(cfg)
    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    splits = exp.get("splits", ["test"])
    generate_base_images = bool(exp.get("generate_base_images", True))
    overwrite_ablated = bool(exp.get("overwrite_ablated", False))
    ablations_cfg = cfg.get("ablations", {})

    for dataset_entry in datasets:
        dataset_name = dataset_entry["name"]
        for chart_type in chart_types:
            run_config = _build_run_config(cfg, dataset_entry, chart_type)
            _ensure_base_images(run_config, generate_base_images)

            for split in splits:
                src_dir = _base_dir_info(cfg, dataset_name, chart_type, split)
                dst_dir = _ablated_dir_info(cfg, dataset_name, chart_type, split)

                if not os.path.isdir(src_dir):
                    raise FileNotFoundError(f"Base chart dir not found: {src_dir}")

                os.makedirs(dst_dir, exist_ok=True)

                files = sorted(f for f in os.listdir(src_dir) if f.endswith(".png"))
                for name in files:
                    src_path = os.path.join(src_dir, name)
                    dst_path = os.path.join(dst_dir, name)
                    if os.path.exists(dst_path) and not overwrite_ablated:
                        continue
                    img = Image.open(src_path).convert("RGB")
                    ablated = apply_ablation(img, chart_type, ablations_cfg)
                    ablated.save(dst_path)

                print(f"{dataset_name} | {chart_type} | {split} -> {dst_dir}")

    print("Ablated images generated.")


def main():
    parser = argparse.ArgumentParser(description="Generate ablated chart images for Experiment 3B")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    generate_ablated_images(cfg)


if __name__ == "__main__":
    main()
