#!/usr/bin/env python
"""
Experiment 7B: Image Post-Processing Pipeline
==============================================
Tests whether applying image post-processing to chart/encoding images
before CNN training improves classification accuracy.

Post-processing methods:
  - Edge detection: Canny, Sobel
  - Contrast enhancement: Histogram Equalization, CLAHE
  - Combined: Edge + CLAHE

This operates on EXISTING chart/encoding images, so it works with
any previously generated images (line charts, GASF, etc.).

Usage
-----
    python scripts/experiment_7b_image_postprocess.py \\
        --config vtbench/config/experiment_7b_postprocess.yaml
"""

import argparse
import gc
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vtbench.data.loader import read_ucr
from vtbench.data.ts_image_encodings import (
    apply_edge_detection, apply_histogram_equalization, apply_clahe,
)
from vtbench.train.factory import get_chart_model
from vtbench.utils.wandb_logger import WandbLogger
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


# ====================================================================
# Post-processing functions
# ====================================================================

# Post-processing functions must be named (not lambda) for Windows
# multiprocessing pickle compatibility with DataLoader num_workers > 0.

def _pp_none(img):
    return img

def _pp_canny(img):
    return apply_edge_detection(np.array(img.convert('L')), 'canny')

def _pp_sobel(img):
    return apply_edge_detection(np.array(img.convert('L')), 'sobel')

def _pp_hist_eq(img):
    return apply_histogram_equalization(np.array(img.convert('L')))

def _pp_clahe(img):
    return apply_clahe(np.array(img.convert('L')))

def _pp_sobel_clahe(img):
    return apply_clahe(
        apply_edge_detection(np.array(img.convert('L')), 'sobel'))

POST_PROCESSORS = {
    'none': _pp_none,
    'canny': _pp_canny,
    'sobel': _pp_sobel,
    'hist_eq': _pp_hist_eq,
    'clahe': _pp_clahe,
    'sobel_clahe': _pp_sobel_clahe,
}


class PostProcessedDataset(torch.utils.data.Dataset):
    """Load images from a directory and apply post-processing."""

    def __init__(self, image_dir, labels, post_fn_name='none',
                 global_indices=None, image_size=128):
        self.image_dir = image_dir
        self.labels = labels
        self.post_fn_name = post_fn_name
        self.post_fn = POST_PROCESSORS[post_fn_name]
        self.global_indices = global_indices or list(range(len(labels)))
        self.image_size = image_size
        self.normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Auto-detect filename prefix from directory contents
        self._file_prefix = self._detect_prefix()

    def _detect_prefix(self):
        """Auto-detect filename prefix from first file in directory."""
        if not os.path.isdir(self.image_dir):
            return "sample_"
        files = [f for f in os.listdir(self.image_dir) if f.endswith('.png')]
        if not files:
            return "sample_"
        # Extract prefix: everything before the last numeric part
        # e.g. "area_chart_color_with_label_21.png" → "area_chart_color_with_label_"
        # e.g. "sample_0.png" → "sample_"
        import re
        m = re.match(r'^(.*?)(\d+)\.png$', files[0])
        if m:
            return m.group(1)
        return "sample_"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        g_idx = self.global_indices[idx]
        path = os.path.join(self.image_dir, f"{self._file_prefix}{g_idx}.png")

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (self.image_size, self.image_size), (0, 0, 0))

        if self.post_fn_name == 'none':
            # Standard processing
            tensor = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])(img)
        else:
            # Apply post-processing (returns grayscale uint8)
            processed = self.post_fn(img)
            if isinstance(processed, np.ndarray):
                if processed.ndim == 2:
                    processed = Image.fromarray(processed, mode='L').convert('RGB')
                else:
                    processed = Image.fromarray(processed, mode='RGB')
            tensor = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ])(processed)

        tensor = self.normalize(tensor)
        return tensor, self.labels[idx]


# ====================================================================
# Training
# ====================================================================

def train_and_evaluate(train_ds, test_ds, model_name, pretrained,
                       lr, num_classes, epochs=100, batch_size=32):
    model = None
    train_loader = None
    test_loader = None
    try:
        safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
        # Disable drop_last when batch_size >= dataset size to avoid empty loader
        use_drop_last = (len(train_ds) > safe_bs)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=safe_bs, shuffle=True, drop_last=use_drop_last,
            num_workers=2, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=min(batch_size, len(test_ds)), shuffle=False,
            num_workers=2, pin_memory=True)

        if len(train_loader) == 0:
            print(f"  WARNING: train_loader empty (n={len(train_ds)}, bs={safe_bs})")
            return 0.0

        model = get_chart_model(model_name, num_classes=num_classes,
                                pretrained=pretrained).to(device)
        # torch.compile disabled — no Triton on Windows
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)
        criterion = nn.CrossEntropyLoss()

        # AMP mixed precision for ~2x throughput
        use_amp = (device == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_acc = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()

            model.eval()
            correct = total = 0
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                for images, labels in test_loader:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    preds = torch.argmax(model(images), dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            acc = correct / total if total > 0 else 0.0
            scheduler.step(total_loss / len(train_loader))

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break

        return best_acc
    except Exception as e:
        import traceback
        print(f"  ERROR in train_and_evaluate: {e}")
        traceback.print_exc()
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
    output_dir = exp.get("output_dir", "results/experiment_7b_postprocess")
    os.makedirs(output_dir, exist_ok=True)

    datasets = exp["datasets"]
    seeds = exp.get("seeds", [42, 123, 7])
    batch_size = cfg.get("training", {}).get("batch_size", 32)
    epochs = cfg.get("training", {}).get("epochs", 100)
    image_size = exp.get("image_size", 128)

    # Source images to post-process
    source_encodings = exp.get("source_encodings", ["line", "gasf", "rp_grayscale"])
    # Post-processing methods
    postprocess_methods = exp.get("postprocess_methods",
                                  ["none", "canny", "sobel", "hist_eq", "clahe", "sobel_clahe"])
    models = exp.get("models", [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False, "lr": 0.001},
        {"name": "resnet18_pt", "chart_model": "resnet18", "pretrained": True, "lr": 0.0005},
    ])

    dataset_root = exp.get("dataset_root", "UCRArchive_2018")
    dataset_ext = exp.get("dataset_ext", "tsv")

    # Chart dir name mapping
    CHART_DIR_MAP = {
        'line': 'line_charts_color_with_label',
        'area': 'area_charts_color_with_label',
        'bar': 'bar_charts_border_color_with_label',
        'scatter': 'scatter_charts_plain_color_with_label',
    }

    wb = WandbLogger(project="vtbench", experiment="7b_postprocess", config=cfg)

    CSV_FIELDS = ["dataset", "source", "postprocess", "model", "seed", "accuracy"]
    rows = []
    completed_runs = set()

    # CSV resume logic
    csv_path = os.path.join(output_dir, "accuracy_postprocess.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing CSV: {csv_path}")
        failed_count = 0
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    acc = float(row.get("accuracy", -1))
                except (ValueError, TypeError):
                    acc = -1
                if acc > 0:
                    completed_runs.add((row["dataset"], row["source"], row["postprocess"],
                                       row["model"], row["seed"]))
                    rows.append(row)
                else:
                    failed_count += 1
        print(f"Loaded {len(completed_runs)} completed runs "
              f"({failed_count} failed runs will be retried)")
    else:
        # Ensure CSV exists with header for incremental appends
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        print(f"Created new CSV: {csv_path}")

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

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
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + len(X_test)))

        for src_enc in source_encodings:
            # Determine image directory
            if src_enc in CHART_DIR_MAP:
                dir_name = CHART_DIR_MAP[src_enc]
            else:
                dir_name = src_enc  # math encoding dir

            base_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images")
            train_dir = os.path.join(base_dir, dir_name, "train")
            test_dir = os.path.join(base_dir, dir_name, "test")

            if not os.path.isdir(train_dir):
                print(f"  SKIP {src_enc}: {train_dir} not found")
                continue

            for pp_method in postprocess_methods:
                print(f"  {src_enc} + {pp_method}")

                for model_cfg in models:
                    for seed in seeds:
                        run_key = (dataset_name, src_enc, pp_method, model_cfg["name"], str(seed))
                        if run_key in completed_runs:
                            print(f"    {model_cfg['name']} seed={seed}: SKIPPED (already completed)")
                            continue

                        torch.manual_seed(seed)
                        np.random.seed(seed)

                        train_ds = PostProcessedDataset(
                            train_dir, list(y_train), pp_method,
                            train_indices, image_size)
                        test_ds = PostProcessedDataset(
                            test_dir, list(y_test), pp_method,
                            test_indices, image_size)

                        acc = train_and_evaluate(
                            train_ds, test_ds,
                            model_cfg["chart_model"], model_cfg["pretrained"],
                            model_cfg["lr"], num_classes, epochs, batch_size)

                        print(f"    {model_cfg['name']} seed={seed}: {acc:.4f}")
                        run_info = {
                            "dataset": dataset_name,
                            "source": src_enc,
                            "postprocess": pp_method,
                            "model": model_cfg["name"],
                            "seed": seed,
                            "accuracy": f"{acc:.4f}",
                        }
                        rows.append(run_info)

                        # Immediately append to CSV (crash-safe)
                        with open(csv_path, "a", newline="", encoding="utf-8") as f:
                            writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                            writer.writerow(run_info)

                        wb.log_run_result(
                            name=f"{dataset_name}/{src_enc}+{pp_method}/{model_cfg['name']}/s{seed}",
                            config=run_info, accuracy=acc,
                        )

                        # GPU cleanup to prevent CUDA driver corruption
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

    # CSV rows are already written incrementally above.
    # Final summary only.
    print(f"\nAll runs complete. Total rows (including resumed): {len(rows)}")

    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment 7B: Image Post-Processing Pipeline\n\n")
        f.write(f"Source encodings: {source_encodings}\n")
        f.write(f"Post-processing: {postprocess_methods}\n")
        f.write(f"Datasets: {len(datasets)}\n\n")

        for src in source_encodings:
            f.write(f"\n## Source: {src}\n\n")
            f.write("| PostProcess | " + " | ".join(m["name"] for m in models) + " |\n")
            f.write("|" + "---|" * (len(models) + 1) + "\n")
            for pp in postprocess_methods:
                vals = []
                for m in models:
                    accs = [float(r["accuracy"]) for r in rows
                            if r["source"] == src and r["postprocess"] == pp
                            and r["model"] == m["name"]]
                    if accs:
                        vals.append(f"{np.mean(accs):.4f}")
                    else:
                        vals.append("N/A")
                f.write(f"| {pp} | " + " | ".join(vals) + " |\n")

    print(f"\nResults: {csv_path}")
    print(f"Summary: {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_experiment(cfg)
