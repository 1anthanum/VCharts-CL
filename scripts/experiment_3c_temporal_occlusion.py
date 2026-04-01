import argparse
import copy
import csv
import os
import random

import numpy as np
import torch
import yaml
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vtbench.train.trainer import train_model
from vtbench.data.loader import create_dataloaders
from vtbench.data.chart_generator import (
    GlobalYRangeCalculator,
    create_line_chart,
    create_area_chart,
    create_bar_chart,
    create_scatter_chart,
)
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


def _ensure_base_images(config):
    base_cfg = copy.deepcopy(config)
    base_cfg["image_generation"].setdefault("generate_images", True)
    base_cfg["image_generation"].setdefault("overwrite_existing", False)
    create_dataloaders(base_cfg)


def _select_indices(labels, preds, count, correct_only, seed):
    indices = np.arange(len(labels))
    if correct_only:
        indices = indices[preds == labels]
    indices = indices.tolist()
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    return indices[:count]


def _overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = np.clip(heatmap, 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    hm = cmap(heatmap)[:, :, :3]
    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), hm_img, alpha)


def _gradcam_time_curve(cam, target_len):
    cam = cam.cpu().numpy()
    if cam.ndim == 2:
        cam2d = cam
    else:
        cam2d = cam.squeeze()
    h, w = cam2d.shape
    # Crop borders to reduce axis/edge artifacts in the 1D projection.
    crop_frac = 0.1
    h_margin = min(int(round(h * crop_frac)), (h - 1) // 2)
    w_margin = min(int(round(w * crop_frac)), (w - 1) // 2)
    if h_margin > 0 or w_margin > 0:
        cam2d = cam2d[h_margin:h - h_margin, w_margin:w - w_margin]
        h, w = cam2d.shape
    x_old = np.linspace(0, target_len - 1, num=w)
    x_new = np.arange(target_len)
    cam_1d = cam2d.mean(axis=0)
    cam_1d = np.interp(x_new, x_old, cam_1d)
    cam_1d = cam_1d - cam_1d.min()
    denom = cam_1d.max() + 1e-6
    cam_1d = cam_1d / denom
    return cam_1d


def _chart_image_from_series(ts, chart_type, chart_cfg, out_path, global_y_range=None):
    color_mode = chart_cfg.get("color_mode", "color")
    label_mode = chart_cfg.get("label_mode", "with_label")
    bar_mode = chart_cfg.get("bar_mode", "border")
    scatter_mode = chart_cfg.get("scatter_mode", "plain")

    if chart_type == "line":
        create_line_chart(ts, out_path, color_mode, label_mode, global_y_range)
    elif chart_type == "area":
        create_area_chart(ts, out_path, color_mode, label_mode, global_y_range)
    elif chart_type == "bar":
        create_bar_chart(ts, out_path, bar_mode, color_mode, label_mode, global_y_range)
    elif chart_type == "scatter":
        create_scatter_chart(ts, out_path, scatter_mode, color_mode, label_mode, global_y_range)
    else:
        raise ValueError(f"Unsupported chart type: {chart_type}")


def _occlusion_curve(ts, model, transform, device, chart_type, chart_cfg, global_y_range, occl_cfg, baseline_logits):
    ts = np.asarray(ts).astype(np.float32).copy()
    if ts.ndim > 1:
        ts = ts.squeeze()
    T = len(ts)
    window_len = max(1, int(T * occl_cfg["window_fraction"]))
    stride = max(1, int(T * occl_cfg["stride_fraction"]))
    # Default to mean fill to avoid introducing zero-value artifacts.
    fill = occl_cfg.get("fill_strategy", "mean")
    score_mode = occl_cfg.get("score", "logit_drop")
    target_class = int(torch.argmax(baseline_logits, dim=1).item())

    if fill == "mean":
        fill_val = float(np.mean(ts))
    elif fill == "median":
        fill_val = float(np.median(ts))
    else:
        fill_val = 0.0

    baseline_logit = float(baseline_logits[0, target_class].item())
    baseline_prob = float(torch.softmax(baseline_logits, dim=1)[0, target_class].item())

    tmp_path = os.path.join(occl_cfg["_tmp_dir"], "occlusion_tmp.png")
    centers = []
    drops = []

    model.eval()
    with torch.no_grad():
        for start in range(0, T - window_len + 1, stride):
            occluded = ts.copy()
            occluded[start:start + window_len] = fill_val
            _chart_image_from_series(
                occluded, chart_type, chart_cfg, tmp_path, global_y_range=global_y_range
            )
            img = Image.open(tmp_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            logits = model(tensor)
            if score_mode == "prob_drop":
                prob = float(torch.softmax(logits, dim=1)[0, target_class].item())
                drop = baseline_prob - prob
            else:
                logit = float(logits[0, target_class].item())
                drop = baseline_logit - logit
            centers.append(start + window_len // 2)
            drops.append(drop)

    centers = np.array(centers, dtype=np.float32)
    drops = np.array(drops, dtype=np.float32)
    if len(centers) == 0:
        return np.zeros(T, dtype=np.float32)
    if len(centers) == 1:
        curve = np.full(T, drops[0], dtype=np.float32)
    else:
        curve = np.interp(np.arange(T), centers, drops).astype(np.float32)

    curve = curve - np.min(curve)
    denom = float(np.max(curve)) + 1e-6
    curve = curve / denom
    return curve


def _corr(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _max_shift_corr(a, b, max_shift):
    # Allow small temporal shifts to reduce sensitivity to off-by-few alignment.
    a = np.asarray(a)
    b = np.asarray(b)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    a = a[:n]
    b = b[:n]
    max_shift = int(max_shift)
    if n <= 1 or max_shift <= 0:
        return _corr(a, b), 0
    max_shift = min(max_shift, n - 1)
    best_corr = float("-inf")
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        if shift < 0:
            a_seg = a[-shift:]
            b_seg = b[:shift]
        elif shift > 0:
            a_seg = a[:-shift]
            b_seg = b[shift:]
        else:
            a_seg = a
            b_seg = b
        corr = _corr(a_seg, b_seg)
        if corr > best_corr:
            best_corr = corr
            best_shift = shift
    return best_corr, best_shift


def _alignment_label(corr):
    if corr >= 0.5:
        return "good"
    if corr >= 0.25:
        return "partial"
    return "poor"


def _set_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _aggregate_alignment(rows):
    grouped = {}
    for row in rows:
        key = (row["dataset"], row["encoding"])
        grouped.setdefault(key, {"avg_corr": [], "samples": []})
        grouped[key]["avg_corr"].append(float(row["avg_corr"]))
        grouped[key]["samples"].append(int(row["samples"]))

    summary = []
    for key, values in grouped.items():
        corr = np.array(values["avg_corr"], dtype=np.float32)
        samples = np.array(values["samples"], dtype=np.float32)
        mean_corr = float(np.mean(corr)) if len(corr) else 0.0
        summary.append(
            {
                "dataset": key[0],
                "encoding": key[1],
                "alignment_quality": _alignment_label(mean_corr),
                "avg_corr_mean": mean_corr,
                "avg_corr_std": float(np.std(corr, ddof=1)) if len(corr) > 1 else 0.0,
                "runs": int(len(corr)),
                "samples_per_run_mean": float(np.mean(samples)) if len(samples) else 0.0,
            }
        )
    return summary


def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_3c")
    os.makedirs(output_dir, exist_ok=True)
    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    samples_per_dataset = int(exp.get("samples_per_dataset", 3))
    correct_only = bool(exp.get("correct_only", True))
    runs = int(exp.get("runs", 1))
    seed = exp.get("seed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rows = []
    alignment_rows = []

    for run_id in range(1, runs + 1):
        run_seed = None if seed is None else int(seed) + run_id - 1
        _set_seeds(run_seed)

        run_dir = os.path.join(output_dir, f"run_{run_id:02d}")
        os.makedirs(run_dir, exist_ok=True)
        tmp_dir = os.path.join(run_dir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        occl_cfg = dict(cfg.get("occlusion", {}))
        occl_cfg["_tmp_dir"] = tmp_dir

        for dataset_entry in datasets:
            dataset_name = dataset_entry["name"]
            for chart_type in chart_types:
                run_config = _build_run_config(cfg, dataset_entry, chart_type)
                _ensure_base_images(run_config)

                model = train_model(run_config).to(device)
                model.eval()

                loaders = create_dataloaders(run_config)
                test_loader = loaders["test"]["chart"]
                dataset = test_loader.dataset

                preds = []
                labels = []
                with torch.no_grad():
                    for images, y in test_loader:
                        images = images.to(device)
                        logits = model(images)
                        pred = torch.argmax(logits, dim=1).cpu().numpy()
                        preds.extend(pred)
                        labels.extend(y.numpy())

                preds = np.array(preds)
                labels = np.array(labels)

                indices = _select_indices(labels, preds, samples_per_dataset, correct_only, run_seed)
                if not indices:
                    print(f"[WARN] No samples selected for {dataset_name} {chart_type}.")
                    continue

                chart_cfg = run_config["chart_branches"]["branch_1"]
                global_y = dataset.uni_global_y
                if global_y is None:
                    global_y = GlobalYRangeCalculator.calculate_global_y_range_univariate(
                        dataset.time_series_data
                    )

                target_layer = find_last_conv_layer(model)
                gradcam = GradCAM(model, target_layer)

                corr_values = []

                for idx in indices:
                    ts = dataset.time_series_data[idx]
                    if isinstance(ts, np.ndarray) and ts.ndim > 1:
                        ts = ts.squeeze(0)
                    ts = np.asarray(ts, dtype=np.float32)

                    filename = dataset._get_image_filename(idx)
                    img_path = os.path.join(dataset.chart_dir, filename)
                    img = Image.open(img_path).convert("RGB")
                    tensor = dataset.transform(img).unsqueeze(0).to(device)

                    # Use the same sample's baseline forward pass for target selection.
                    with torch.no_grad():
                        baseline_logits = model(tensor)
                    target_class = int(torch.argmax(baseline_logits, dim=1).item())
                    cam, _ = gradcam(tensor, class_idx=target_class)
                    cam_curve = _gradcam_time_curve(cam, len(ts))

                    occl_curve = _occlusion_curve(
                        ts,
                        model,
                        dataset.transform,
                        device,
                        chart_type,
                        chart_cfg,
                        global_y,
                        occl_cfg,
                        baseline_logits,
                    )

                    max_shift = int(round(len(ts) * 0.05))
                    corr, shift = _max_shift_corr(cam_curve, occl_curve, max_shift)
                    label = _alignment_label(corr)
                    corr_values.append(corr)

                    overlay = _overlay_heatmap(img, cam.cpu().numpy(), alpha=0.5)

                    sample_dir = os.path.join(run_dir, dataset_name, chart_type)
                    os.makedirs(sample_dir, exist_ok=True)
                    out_path = os.path.join(sample_dir, f"sample_{idx}.png")

                    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
                    axes[0].imshow(overlay)
                    axes[0].set_title(f"{dataset_name} {chart_type}")
                    axes[0].axis("off")
                    axes[1].plot(occl_curve, label="occlusion", color="#d62728")
                    axes[1].plot(cam_curve, label="gradcam", color="#1f77b4")
                    axes[1].set_title(f"corr={corr:.2f} shift={shift} ({label})")
                    axes[1].set_xlabel("time index")
                    axes[1].set_ylabel("normalized")
                    axes[1].legend(frameon=False)
                    fig.tight_layout()
                    fig.savefig(out_path, dpi=150)
                    plt.close(fig)

                    sample_rows.append(
                        {
                            "run_id": run_id,
                            "seed": run_seed,
                            "dataset": dataset_name,
                            "encoding": chart_type,
                            "index": int(idx),
                            "label": int(labels[idx]),
                            "prediction": int(preds[idx]),
                            "corr": f"{corr:.4f}",
                            "alignment": label,
                            "output_path": out_path,
                        }
                    )

                gradcam.clear()

                avg_corr = float(np.mean(corr_values)) if corr_values else 0.0
                if corr_values:
                    labels_list = [_alignment_label(c) for c in corr_values]
                    majority = max(set(labels_list), key=labels_list.count)
                else:
                    majority = "poor"

                alignment_rows.append(
                    {
                        "run_id": run_id,
                        "seed": run_seed,
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "alignment_quality": majority,
                        "avg_corr": f"{avg_corr:.4f}",
                        "samples": len(corr_values),
                    }
                )

    samples_csv = os.path.join(output_dir, "sample_alignment_runs.csv")
    with open(samples_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_id",
                "seed",
                "dataset",
                "encoding",
                "index",
                "label",
                "prediction",
                "corr",
                "alignment",
                "output_path",
            ],
        )
        writer.writeheader()
        writer.writerows(sample_rows)

    alignment_csv = os.path.join(output_dir, "alignment_table_runs.csv")
    with open(alignment_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "seed", "dataset", "encoding", "alignment_quality", "avg_corr", "samples"],
        )
        writer.writeheader()
        writer.writerows(alignment_rows)

    summary = _aggregate_alignment(alignment_rows)
    summary_csv = os.path.join(output_dir, "alignment_table_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "encoding",
                "alignment_quality",
                "avg_corr_mean",
                "avg_corr_std",
                "runs",
                "samples_per_run_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    summary_alias = os.path.join(output_dir, "alignment_table.csv")
    with open(summary_alias, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "encoding",
                "alignment_quality",
                "avg_corr_mean",
                "avg_corr_std",
                "runs",
                "samples_per_run_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)

    print(f"Saved outputs to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3C: Temporal Occlusion vs Grad-CAM")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
