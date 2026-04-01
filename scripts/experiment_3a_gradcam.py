import argparse
import copy
import csv
import os

import numpy as np
import torch
from PIL import Image
import yaml
import matplotlib.cm as cm

from vtbench.train.trainer import train_model
from vtbench.data.loader import create_dataloaders
from vtbench.utils.gradcam import GradCAM, find_last_conv_layer


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
                    "category": exp.get("dataset_category", {}).get(name, "unspecified"),
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
                    "category": item.get("category", exp.get("dataset_category", {}).get(name, "unspecified")),
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


def _ensure_base_images(config):
    base_cfg = copy.deepcopy(config)
    base_cfg["image_generation"].setdefault("generate_images", True)
    base_cfg["image_generation"].setdefault("overwrite_existing", False)
    create_dataloaders(base_cfg)


def _overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = heatmap.numpy()
    heatmap = np.clip(heatmap, 0.0, 1.0)
    heatmap_img = cm.get_cmap("jet")(heatmap)
    heatmap_img = Image.fromarray((heatmap_img[:, :, :3] * 255).astype(np.uint8))
    heatmap_img = heatmap_img.resize(image.size, resample=Image.BILINEAR)
    return Image.blend(image.convert("RGB"), heatmap_img, alpha)


def _collect_predictions(model, loader, device):
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            images, y = batch
            images = images.to(device)
            y = y.to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            preds.extend(pred.cpu().numpy())
            labels.extend(y.cpu().numpy())
    return np.array(preds), np.array(labels)


def _sample_indices(preds, labels, samples_per_class, rng):
    indices = np.arange(len(labels))
    picks = {}
    for cls in [0, 1]:
        cls_idx = indices[labels == cls]
        correct_idx = cls_idx[preds[cls_idx] == cls]
        incorrect_idx = cls_idx[preds[cls_idx] != cls]

        correct_idx = rng.permutation(correct_idx)
        incorrect_idx = rng.permutation(incorrect_idx)

        picks[(cls, "correct")] = correct_idx[:samples_per_class].tolist()
        picks[(cls, "incorrect")] = incorrect_idx[:samples_per_class].tolist()
    return picks


def _grid_from_paths(paths_by_row, cell_size, pad, bgcolor):
    rows = len(paths_by_row)
    cols = max(len(row) for row in paths_by_row) if rows else 0
    if rows == 0 or cols == 0:
        return None

    width = cols * cell_size + (cols + 1) * pad
    height = rows * cell_size + (rows + 1) * pad
    grid = Image.new("RGB", (width, height), bgcolor)

    for r, row in enumerate(paths_by_row):
        for c, path in enumerate(row):
            x0 = pad + c * (cell_size + pad)
            y0 = pad + r * (cell_size + pad)
            if path is None or not os.path.isfile(path):
                continue
            img = Image.open(path).convert("RGB").resize((cell_size, cell_size), Image.BILINEAR)
            grid.paste(img, (x0, y0))
    return grid


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_3a")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    samples_per_class = int(exp.get("samples_per_class", 10))
    cam_target = exp.get("cam_target", "pred")
    if cam_target not in ("pred", "true"):
        raise ValueError(f"cam_target must be 'pred' or 'true', got: {cam_target}")
    seed = exp.get("seed")
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    overlay_alpha = float(exp.get("overlay_alpha", 0.5))
    cell_size = int(exp.get("grid_cell_size", 160))
    pad = int(exp.get("grid_padding", 6))
    bgcolor = tuple(exp.get("grid_bgcolor", [255, 255, 255]))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_rows = []
    table_rows = []
    notes_rows = []

    for dataset_entry in datasets:
        dataset_name = dataset_entry["name"]
        category = dataset_entry.get("category", "unspecified")
        dataset_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)

        grid_paths = {"class0": {}, "class1": {}}

        for chart_type in chart_types:
            run_config = _build_run_config(cfg, dataset_entry, chart_type)
            _ensure_base_images(run_config)

            model = train_model(run_config)
            model = model.to(device)

            loaders = create_dataloaders(run_config)
            test_loader = loaders["test"]["chart"]
            dataset = test_loader.dataset

            preds, labels = _collect_predictions(model, test_loader, device)
            picks = _sample_indices(preds, labels, samples_per_class, rng)

            if len(np.unique(labels)) < 2:
                print(f"[WARN] {dataset_name} has <2 classes after parsing; skipping.")
                continue

            model.eval()
            target_layer = find_last_conv_layer(model)
            gradcam = GradCAM(model, target_layer)

            for cls in [0, 1]:
                for correctness in ["correct", "incorrect"]:
                    indices = picks[(cls, correctness)]
                    if len(indices) < samples_per_class:
                        print(
                            f"[WARN] {dataset_name}/{chart_type} class {cls} "
                            f"{correctness} has {len(indices)}/{samples_per_class} samples."
                        )
                    out_dir = os.path.join(
                        dataset_dir, chart_type, f"class_{cls}", correctness
                    )
                    os.makedirs(out_dir, exist_ok=True)

                    for idx in indices:
                        filename = dataset._get_image_filename(idx)
                        img_path = os.path.join(dataset.chart_dir, filename)
                        img = Image.open(img_path).convert("RGB")
                        tensor = dataset.transform(img).unsqueeze(0).to(device)

                        target_cls = int(preds[idx]) if cam_target == "pred" else int(labels[idx])
                        cam, logits = gradcam(tensor, class_idx=target_cls)
                        overlay = _overlay_heatmap(img, cam, alpha=overlay_alpha)

                        out_name = f"idx_{idx}_pred_{target_cls}.png"
                        out_path = os.path.join(out_dir, out_name)
                        overlay.save(out_path)

                        manifest_rows.append(
                            {
                                "dataset": dataset_name,
                                "category": category,
                                "encoding": chart_type,
                                "class_label": int(labels[idx]),
                                "prediction": int(preds[idx]),
                                "correctness": correctness,
                                "index": int(idx),
                                "image_path": img_path,
                                "overlay_path": out_path,
                                "cam_target": cam_target,
                                "cam_class": target_cls,
                            }
                        )

                        grid_key = "class0" if cls == 0 else "class1"
                        grid_paths.setdefault(grid_key, {}).setdefault(chart_type, []).append(out_path)

            gradcam.clear()

        # Build grids per dataset
        for cls_key, grid_name in [("class0", "grid_class0.png"), ("class1", "grid_class1.png")]:
            paths_by_row = []
            for chart_type in chart_types:
                row = grid_paths.get(cls_key, {}).get(chart_type, [])
                needed = samples_per_class * 2
                row = row[:needed]
                if len(row) < needed:
                    row = row + [None] * (needed - len(row))
                paths_by_row.append(row)
            grid_img = _grid_from_paths(paths_by_row, cell_size, pad, bgcolor)
            if grid_img:
                grid_img.save(os.path.join(dataset_dir, grid_name))

        notes_rows.append(
            f"{dataset_name} ({category}): [Fill 2-3 sentences after inspecting grids]\n"
        )
        table_rows.append(
            {
                "dataset": dataset_name,
                "category": category,
                "encoding_consistency": "TBD",
                "suspected_artifact_influence": "TBD",
            }
        )

    # Write manifest
    manifest_path = os.path.join(output_dir, "gradcam_manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "category",
                "encoding",
                "class_label",
                "prediction",
                "correctness",
                "index",
                "image_path",
                "overlay_path",
                "cam_target",
                "cam_class",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write table template
    table_path = os.path.join(output_dir, "encoding_consistency_table.csv")
    with open(table_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "category",
                "encoding_consistency",
                "suspected_artifact_influence",
            ],
        )
        writer.writeheader()
        writer.writerows(table_rows)

    # Write notes template
    notes_path = os.path.join(output_dir, "dataset_notes.md")
    with open(notes_path, "w") as f:
        f.write("# Experiment 3A Notes\n\n")
        for line in notes_rows:
            f.write(line)

    print(f"Saved Grad-CAM overlays and grids to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3A: Grad-CAM Analysis")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
