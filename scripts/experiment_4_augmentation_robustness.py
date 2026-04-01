"""
Experiment 4: Augmentation Robustness
======================================
Evaluates how deterministic image-domain perturbations affect:
  1. Classification accuracy (performance robustness)
  2. Grad-CAM / occlusion alignment (interpretability robustness)
across chart encodings (line, area, bar, scatter).

Uses fixed, pre-trained models — augmentations are applied at test
time only.

Usage
-----
    python scripts/experiment_4_augmentation_robustness.py \
        --config vtbench/config/experiment_4_augmentation_robustness.yaml
"""

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
from vtbench.data.chart_generator import GlobalYRangeCalculator
from vtbench.utils.gradcam import GradCAM, find_last_conv_layer
from vtbench.utils.augmentations import apply_augmentation, augmentation_label


# ====================================================================
# Helpers (shared patterns from experiment_3c)
# ====================================================================

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
            entries.append({
                "name": name,
                "train_path": train_path,
                "test_path": test_path,
                "format": exp.get("dataset_format", "ucr"),
            })
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
            entries.append({
                "name": name,
                "train_path": train_path,
                "test_path": test_path,
                "format": item.get("format", exp.get("dataset_format", "ucr")),
            })
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


def _set_seeds(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _select_indices(labels, preds, count, correct_only, seed):
    indices = np.arange(len(labels))
    if correct_only:
        indices = indices[preds == labels]
    indices = indices.tolist()
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    return indices[:count]


# ====================================================================
# Grad-CAM and occlusion helpers (from experiment_3c, kept consistent)
# ====================================================================

def _gradcam_time_curve(cam, target_len):
    cam = cam.cpu().numpy()
    if cam.ndim == 2:
        cam2d = cam
    else:
        cam2d = cam.squeeze()
    h, w = cam2d.shape
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


def _corr(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _max_shift_corr(a, b, max_shift):
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


def _overlay_heatmap(image, heatmap, alpha=0.5):
    heatmap = np.clip(heatmap, 0.0, 1.0)
    cmap = plt.get_cmap("jet")
    hm = cmap(heatmap)[:, :, :3]
    hm_img = Image.fromarray((hm * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    return Image.blend(image.convert("RGB"), hm_img, alpha)


# Occlusion (chart-image-domain): re-render chart with mean-fill occlusion
# Imported chart creation functions for on-the-fly re-rendering
from vtbench.data.chart_generator import (
    create_line_chart,
    create_area_chart,
    create_bar_chart,
    create_scatter_chart,
)


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


def _occlusion_curve(ts, model, transform, device, chart_type, chart_cfg,
                     global_y_range, occl_cfg, baseline_logits):
    ts = np.asarray(ts).astype(np.float32).copy()
    if ts.ndim > 1:
        ts = ts.squeeze()
    T = len(ts)
    window_len = max(1, int(T * occl_cfg["window_fraction"]))
    stride = max(1, int(T * occl_cfg["stride_fraction"]))
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
            # Free references to avoid matplotlib/PIL memory leaks
            del img, tensor, logits

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


# ====================================================================
# Accuracy evaluation (clean and augmented)
# ====================================================================

def _evaluate_accuracy(model, test_loader, device, augment_fn=None):
    """Evaluate accuracy on a test loader, optionally applying an
    augmentation to each image before inference.

    Parameters
    ----------
    augment_fn : callable or None
        If provided, called as augment_fn(pil_image) -> pil_image for
        each sample.  When None the loader's built-in transform is used
        directly (clean baseline).
    """
    model.eval()
    correct = 0
    total = 0

    if augment_fn is not None:
        # We need to intercept the raw PIL images and apply the
        # augmentation before the standard transform pipeline.
        dataset = test_loader.dataset
        transform = dataset.transform
        with torch.no_grad():
            for idx in range(len(dataset)):
                # Access the raw image from the dataset
                img_path = os.path.join(
                    dataset.chart_dir,
                    dataset._get_image_filename(idx),
                )
                pil_img = Image.open(img_path).convert("RGB")
                pil_img = augment_fn(pil_img)
                tensor = transform(pil_img).unsqueeze(0).to(device)
                label = dataset.labels[idx]
                logits = model(tensor)
                pred = int(torch.argmax(logits, dim=1).item())
                correct += int(pred == label)
                total += 1
    else:
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels.to(device)).sum().item()
                total += labels.size(0)

    return correct / total if total > 0 else 0.0


# ====================================================================
# Alignment evaluation on augmented images
# ====================================================================

def _evaluate_alignment_augmented(
    model, dataset, indices, transform, device,
    chart_type, chart_cfg, global_y_range, occl_cfg,
    augment_fn, gradcam_obj,
):
    """Compute alignment metrics for selected samples under augmentation.

    Returns list of dicts with per-sample correlation, shift, label.
    """
    results = []
    for idx in indices:
        ts = dataset.time_series_data[idx]
        if isinstance(ts, np.ndarray) and ts.ndim > 1:
            ts = ts.squeeze(0)
        ts = np.asarray(ts, dtype=np.float32)

        # Load original image and apply augmentation
        filename = dataset._get_image_filename(idx)
        img_path = os.path.join(dataset.chart_dir, filename)
        pil_img = Image.open(img_path).convert("RGB")
        aug_img = augment_fn(pil_img)

        tensor = transform(aug_img).unsqueeze(0).to(device)

        # Baseline logits from augmented image (consistent target)
        with torch.no_grad():
            baseline_logits = model(tensor)
        target_class = int(torch.argmax(baseline_logits, dim=1).item())

        # Grad-CAM on augmented image
        cam, _ = gradcam_obj(tensor, class_idx=target_class)
        cam_curve = _gradcam_time_curve(cam, len(ts))

        # Occlusion curve (signal domain, same target class)
        occl_curve = _occlusion_curve(
            ts, model, transform, device,
            chart_type, chart_cfg, global_y_range,
            occl_cfg, baseline_logits,
        )

        max_shift = int(round(len(ts) * 0.05))
        corr, shift = _max_shift_corr(cam_curve, occl_curve, max_shift)
        label = _alignment_label(corr)

        results.append({
            "index": int(idx),
            "corr": corr,
            "shift": shift,
            "alignment": label,
            "cam_curve": cam_curve,
            "occl_curve": occl_curve,
            "aug_img": aug_img,
        })
    return results


# ====================================================================
# Figure generation
# ====================================================================

def _save_comparison_figure(
    clean_img, aug_img, cam_curve, occl_curve,
    corr, shift, label, dataset_name, chart_type, aug_label, out_path,
):
    """Save a side-by-side figure: clean | augmented + alignment curves."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    axes[0].imshow(clean_img)
    axes[0].set_title("Clean")
    axes[0].axis("off")
    axes[1].imshow(aug_img)
    axes[1].set_title(f"Aug: {aug_label}")
    axes[1].axis("off")
    axes[2].plot(occl_curve, label="occlusion", color="#d62728")
    axes[2].plot(cam_curve, label="gradcam", color="#1f77b4")
    axes[2].set_title(f"corr={corr:.2f} shift={shift} ({label})")
    axes[2].set_xlabel("time index")
    axes[2].set_ylabel("normalized")
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle(f"{dataset_name} / {chart_type}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ====================================================================
# Main experiment loop
# ====================================================================

def run_experiment(cfg):
    exp = cfg["experiment"]
    output_dir = exp.get("output_dir", "results/experiment_4_augmentation_robustness")
    os.makedirs(output_dir, exist_ok=True)

    chart_types = exp.get("chart_types", ["line", "area", "bar", "scatter"])
    datasets = _dataset_entries(cfg)
    samples_per_dataset = int(exp.get("samples_per_dataset", 5))
    correct_only = bool(exp.get("correct_only", True))
    seed = exp.get("seed", 42)

    aug_specs = cfg.get("augmentations", [])
    occl_cfg = dict(cfg.get("occlusion", {}))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Flatten augmentation specs into (type, params, label) triples
    aug_combos = []
    for spec in aug_specs:
        aug_type = spec["type"]
        for params in spec.get("levels", [{}]):
            lbl = augmentation_label(aug_type, params)
            aug_combos.append((aug_type, dict(params), lbl))

    # Result collectors
    accuracy_rows = []
    alignment_rows = []

    # Mode flag: skip alignment if --accuracy-only CLI flag or env var
    accuracy_only = bool(exp.get("accuracy_only", False))

    _set_seeds(seed)

    for dataset_entry in datasets:
        dataset_name = dataset_entry["name"]
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 60}")

        for chart_type in chart_types:
            print(f"\n  Encoding: {chart_type}")

            run_config = _build_run_config(cfg, dataset_entry, chart_type)
            _ensure_base_images(run_config)

            # Train model (or load cached)
            _set_seeds(seed)
            model = train_model(run_config).to(device)
            model.eval()

            loaders = create_dataloaders(run_config)
            test_loader = loaders["test"]["chart"]
            dataset_obj = test_loader.dataset

            # Compute clean baseline accuracy
            clean_acc = _evaluate_accuracy(model, test_loader, device, augment_fn=None)
            print(f"    Clean accuracy: {clean_acc:.4f}")

            accuracy_rows.append({
                "dataset": dataset_name,
                "encoding": chart_type,
                "augmentation": "clean",
                "accuracy": f"{clean_acc:.4f}",
                "delta_accuracy": "0.0000",
            })

            if not accuracy_only:
                # Collect predictions for sample selection
                preds_list = []
                labels_list = []
                with torch.no_grad():
                    for images, labels in test_loader:
                        images = images.to(device)
                        logits = model(images)
                        pred = torch.argmax(logits, dim=1).cpu().numpy()
                        preds_list.extend(pred)
                        labels_list.extend(labels.numpy())
                preds_arr = np.array(preds_list)
                labels_arr = np.array(labels_list)

                indices = _select_indices(
                    labels_arr, preds_arr, samples_per_dataset, correct_only, seed
                )
                if not indices:
                    print(f"    [WARN] No samples selected for {dataset_name}/{chart_type}")
                    # Still continue with augmentation accuracy below

                chart_cfg = run_config["chart_branches"]["branch_1"]
                global_y = dataset_obj.uni_global_y
                if global_y is None:
                    global_y = GlobalYRangeCalculator.calculate_global_y_range_univariate(
                        dataset_obj.time_series_data
                    )

                if indices:
                    # Clean alignment baseline
                    tmp_dir = os.path.join(output_dir, "tmp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    clean_occl_cfg = dict(occl_cfg)
                    clean_occl_cfg["_tmp_dir"] = tmp_dir

                    target_layer = find_last_conv_layer(model)
                    gradcam = GradCAM(model, target_layer)

                    clean_align = _evaluate_alignment_augmented(
                        model, dataset_obj, indices, dataset_obj.transform, device,
                        chart_type, chart_cfg, global_y, clean_occl_cfg,
                        augment_fn=lambda img: img,  # identity
                        gradcam_obj=gradcam,
                    )
                    clean_corrs = [r["corr"] for r in clean_align]
                    clean_avg_corr = float(np.mean(clean_corrs)) if clean_corrs else 0.0
                    clean_avg_shift = float(np.mean([abs(r["shift"]) for r in clean_align])) if clean_align else 0.0

                    alignment_rows.append({
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "augmentation": "clean",
                        "avg_corr": f"{clean_avg_corr:.4f}",
                        "avg_abs_shift": f"{clean_avg_shift:.2f}",
                        "alignment_label": _alignment_label(clean_avg_corr),
                        "n_good": sum(1 for r in clean_align if r["alignment"] == "good"),
                        "n_partial": sum(1 for r in clean_align if r["alignment"] == "partial"),
                        "n_poor": sum(1 for r in clean_align if r["alignment"] == "poor"),
                    })

                    gradcam.clear()

            # ---- Augmentation sweep ----
            for aug_type, aug_params, aug_lbl in aug_combos:
                def _augment(img, _at=aug_type, _ap=aug_params):
                    return apply_augmentation(img, _at, _ap)

                aug_acc = _evaluate_accuracy(
                    model, test_loader, device, augment_fn=_augment
                )
                delta = aug_acc - clean_acc
                print(f"    {aug_lbl:>20s}: acc={aug_acc:.4f}  Δ={delta:+.4f}")

                accuracy_rows.append({
                    "dataset": dataset_name,
                    "encoding": chart_type,
                    "augmentation": aug_lbl,
                    "accuracy": f"{aug_acc:.4f}",
                    "delta_accuracy": f"{delta:.4f}",
                })

                if not accuracy_only and indices:
                    # Alignment under augmentation
                    target_layer_aug = find_last_conv_layer(model)
                    gradcam_aug = GradCAM(model, target_layer_aug)

                    aug_occl_cfg = dict(occl_cfg)
                    aug_occl_cfg["_tmp_dir"] = tmp_dir

                    aug_align = _evaluate_alignment_augmented(
                        model, dataset_obj, indices, dataset_obj.transform, device,
                        chart_type, chart_cfg, global_y, aug_occl_cfg,
                        augment_fn=_augment,
                        gradcam_obj=gradcam_aug,
                    )
                    aug_corrs = [r["corr"] for r in aug_align]
                    avg_corr = float(np.mean(aug_corrs)) if aug_corrs else 0.0
                    avg_shift = float(np.mean([abs(r["shift"]) for r in aug_align])) if aug_align else 0.0

                    alignment_rows.append({
                        "dataset": dataset_name,
                        "encoding": chart_type,
                        "augmentation": aug_lbl,
                        "avg_corr": f"{avg_corr:.4f}",
                        "avg_abs_shift": f"{avg_shift:.2f}",
                        "alignment_label": _alignment_label(avg_corr),
                        "n_good": sum(1 for r in aug_align if r["alignment"] == "good"),
                        "n_partial": sum(1 for r in aug_align if r["alignment"] == "partial"),
                        "n_poor": sum(1 for r in aug_align if r["alignment"] == "poor"),
                    })

                    # Save representative figure for the first sample
                    if aug_align:
                        r0 = aug_align[0]
                        idx0 = r0["index"]
                        fname = dataset_obj._get_image_filename(idx0)
                        clean_img = Image.open(
                            os.path.join(dataset_obj.chart_dir, fname)
                        ).convert("RGB")

                        fig_dir = os.path.join(
                            output_dir, "figures", dataset_name, chart_type
                        )
                        os.makedirs(fig_dir, exist_ok=True)
                        _save_comparison_figure(
                            clean_img, r0["aug_img"],
                            r0["cam_curve"], r0["occl_curve"],
                            r0["corr"], r0["shift"], r0["alignment"],
                            dataset_name, chart_type, aug_lbl,
                            os.path.join(fig_dir, f"{aug_lbl}_sample{idx0}.png"),
                        )

                    gradcam_aug.clear()
                    # Free alignment results to reduce memory
                    del aug_align

            # Free model after processing all augmentations for this encoding
            import gc
            del model, loaders, test_loader
            gc.collect()

    # ================================================================
    # Write summary CSVs
    # ================================================================

    acc_csv = os.path.join(output_dir, "accuracy_robustness.csv")
    with open(acc_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "encoding", "augmentation", "accuracy", "delta_accuracy"],
        )
        writer.writeheader()
        writer.writerows(accuracy_rows)

    align_csv = os.path.join(output_dir, "alignment_robustness.csv")
    with open(align_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "encoding", "augmentation",
                "avg_corr", "avg_abs_shift", "alignment_label",
                "n_good", "n_partial", "n_poor",
            ],
        )
        writer.writeheader()
        writer.writerows(alignment_rows)

    # ---- Generate summary markdown ----
    _write_summary_md(output_dir, accuracy_rows, alignment_rows)

    print(f"\nAll outputs saved to {output_dir}/")
    print(f"  - accuracy_robustness.csv")
    print(f"  - alignment_robustness.csv")
    print(f"  - summary.md")
    print(f"  - figures/")


def _write_summary_md(output_dir, accuracy_rows, alignment_rows):
    """Generate a markdown summary table for quick inspection."""
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write("# Experiment 4: Augmentation Robustness — Summary\n\n")

        f.write("## Accuracy Robustness\n\n")
        f.write("| Dataset | Encoding | Augmentation | Accuracy | ΔAccuracy |\n")
        f.write("|---------|----------|-------------|----------|------------|\n")
        for row in accuracy_rows:
            f.write(
                f"| {row['dataset']} | {row['encoding']} | {row['augmentation']} "
                f"| {row['accuracy']} | {row['delta_accuracy']} |\n"
            )

        f.write("\n## Alignment Robustness\n\n")
        f.write("| Dataset | Encoding | Augmentation | Avg Corr | Avg |Shift| | Label | Good | Partial | Poor |\n")
        f.write("|---------|----------|-------------|----------|-----------|-------|------|---------|------|\n")
        for row in alignment_rows:
            f.write(
                f"| {row['dataset']} | {row['encoding']} | {row['augmentation']} "
                f"| {row['avg_corr']} | {row['avg_abs_shift']} | {row['alignment_label']} "
                f"| {row['n_good']} | {row['n_partial']} | {row['n_poor']} |\n"
            )


# ====================================================================
# CLI entry point
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: Augmentation Robustness"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
