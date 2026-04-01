"""
Unified image dataset classes for loading pre-generated chart and encoding images.

These replace the duplicated EncodingImageDataset and ChartImageDataset that
were copy-pasted across experiment_6a, experiment_8a, and others.

Usage:
    from vtbench.data.image_dataset import EncodingImageDataset, ChartImageDataset

Both classes expect images to be pre-generated under CHART_IMAGE_ROOT.
"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")


class EncodingImageDataset(Dataset):
    """Load pre-generated encoding images (GASF, MTF, RP, CWT, phase, RGB stacks, etc.).

    Directory structure expected:
        {CHART_IMAGE_ROOT}/{dataset_name}_images/{encoding_dir}/{split}/sample_{idx}.png

    Args:
        dataset_name: UCR dataset name (e.g., "ECG5000")
        split: "train", "val", or "test"
        encoding_dir: Subdirectory name for this encoding (e.g., "gasf", "rgb_gasf_gadf_mtf")
        global_indices: Array of global sample indices
        labels: Array of integer labels
        transform: torchvision transform to apply
    """

    def __init__(self, dataset_name, split, encoding_dir, global_indices,
                 labels, transform=None):
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform
        self.img_dir = os.path.join(
            CHART_IMAGE_ROOT, f"{dataset_name}_images", encoding_dir, split,
        )
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Encoding images not found: {self.img_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        img_path = os.path.join(self.img_dir, f"sample_{gidx}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            # Corrupt/missing image — return a blank placeholder and warn once
            if not getattr(self, '_warned_corrupt', False):
                print(f"  WARNING: corrupt image {img_path}, using blank placeholder")
                self._warned_corrupt = True
            img = Image.new("RGB", (128, 128), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


class ChartImageDataset(Dataset):
    """Load pre-generated chart images (line, area, bar, scatter).

    Directory structure expected:
        {CHART_IMAGE_ROOT}/{dataset_name}_images/{chart_type}_charts_{color}_{label}/{split}/

    File naming:
        {chart_type}_chart[_{variant}]_{color}_{label}_{idx}.png

    Args:
        dataset_name: UCR dataset name
        split: "train", "val", or "test"
        chart_type: "line", "area", "bar", or "scatter"
        color_mode: "color" or "monochrome"
        label_mode: "with_label" or "without_label"
        global_indices: Array of global sample indices
        labels: Array of integer labels
        transform: torchvision transform to apply
        bar_mode: "border" or "fill" (only for bar charts)
        scatter_mode: "plain" or other variants (only for scatter charts)
    """

    def __init__(self, dataset_name, split, chart_type, color_mode,
                 label_mode, global_indices, labels, transform=None,
                 bar_mode="border", scatter_mode="plain"):
        self.labels = labels
        self.global_indices = global_indices
        self.transform = transform
        self.chart_type = chart_type

        base = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images")
        if chart_type == "bar":
            self.img_dir = os.path.join(
                base, f"bar_charts_{bar_mode}_{color_mode}_{label_mode}", split)
        elif chart_type == "scatter":
            self.img_dir = os.path.join(
                base, f"scatter_charts_{scatter_mode}_{color_mode}_{label_mode}", split)
        else:
            self.img_dir = os.path.join(
                base, f"{chart_type}_charts_{color_mode}_{label_mode}", split)

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Chart images not found: {self.img_dir}")

        # Filename components
        self._prefix = {
            "line": "line_chart",
            "area": "area_chart",
            "bar": f"bar_chart_{bar_mode}",
            "scatter": f"scatter_chart_{scatter_mode}",
        }[chart_type]
        self._suffix = f"_{color_mode}_{label_mode}"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        fname = f"{self._prefix}{self._suffix}_{gidx}.png"
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label
