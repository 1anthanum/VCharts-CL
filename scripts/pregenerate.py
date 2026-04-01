#!/usr/bin/env python
"""
Pre-generate ALL image encodings for all datasets.
==================================================
Generates: chart images (line/area/bar/scatter) + mathematical encodings
(GASF/GADF/MTF/RP/CWT/STFT) + RGB stacks + colormapped variants.

This script is designed to run on LOCAL CPU overnight before uploading
to cloud storage (R2). Cloud GPU time is expensive; image generation is
free on local hardware.

Output structure:
    {CHART_IMAGE_ROOT}/{dataset}_images/
        line_charts_color_with_label/{split}/
        area_charts_color_with_label/{split}/
        bar_charts_border_color_with_label/{split}/
        scatter_charts_plain_color_with_label/{split}/
        gasf/{split}/
        gadf/{split}/
        mtf_8/{split}/
        mtf_16/{split}/
        rp/{split}/
        rp_grayscale/{split}/
        cwt/{split}/
        stft/{split}/
        rgb_gasf_gadf_mtf/{split}/
        rgb_rp_cwt_stft/{split}/
        gasf_viridis/{split}/
        gadf_plasma/{split}/
        rp_grayscale_inferno/{split}/

Usage:
    # Generate all encodings for core 5 datasets
    python scripts/pregenerate_all_encodings.py

    # Generate for all 32 datasets
    python scripts/pregenerate_all_encodings.py --all-datasets

    # Only mathematical encodings (skip chart images)
    python scripts/pregenerate_all_encodings.py --skip-charts

    # Only specific encodings
    python scripts/pregenerate_all_encodings.py --encodings gasf gadf mtf rp

    # With colormaps
    python scripts/pregenerate_all_encodings.py --colormaps viridis plasma

    # Specific image size
    python scripts/pregenerate_all_encodings.py --image-size 224
"""

import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtbench.data.loader import read_ucr
from vtbench.data.ts_image_encodings import (
    ENCODING_REGISTRY, RGB_STACK_PRESETS,
    get_encoding, get_rgb_stack,
    save_encoding_image, apply_colormap, apply_edge_detection,
    preprocess_ts, PREPROCESS_OPTIONS,
)

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")

# Dataset groups
CORE_DATASETS = [
    "SyntheticControl", "GunPoint", "CBF", "Trace", "ECG5000",
]

EXTENDED_DATASETS = CORE_DATASETS + [
    "FordA", "TwoPatterns", "Wafer", "BeetleFly", "TwoLeadECG",
    "Coffee", "Lightning2", "ECG200", "Computers", "Yoga",
]

ALL_DATASETS = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "CBF",
    "ChlorineConcentration", "Coffee", "Computers", "CricketX", "CricketY",
    "CricketZ", "ECG200", "ECG5000", "Earthquakes", "FaceAll",
    "Fish", "FordA", "FordB", "GunPoint", "Herring",
    "ItalyPowerDemand", "Lightning2", "SwedishLeaf", "SyntheticControl",
    "ToeSegmentation1", "Trace", "TwoLeadECG", "TwoPatterns",
    "Wafer", "Wine", "WordSynonyms", "Yoga",
]

CHART_TYPES = ["line", "area", "bar", "scatter"]

DEFAULT_ENCODINGS = list(ENCODING_REGISTRY.keys())
DEFAULT_RGB_PRESETS = list(RGB_STACK_PRESETS.keys())
DEFAULT_COLORMAPS = ["viridis", "plasma", "inferno"]
DEFAULT_EDGE_METHODS = ["canny", "sobel"]


def load_dataset(dataset_name):
    """Load train+test data for a dataset."""
    for ext in [".tsv", ".ts"]:
        train_path = f"UCRArchive_2018/{dataset_name}/{dataset_name}_TRAIN{ext}"
        test_path = f"UCRArchive_2018/{dataset_name}/{dataset_name}_TEST{ext}"
        if os.path.exists(train_path):
            X_train, y_train = read_ucr(train_path)
            X_test, y_test = read_ucr(test_path)
            return X_train, y_train, X_test, y_test
    return None, None, None, None


def generate_encoding_images(dataset_name, X, y, split, encoding_name,
                              image_size, overwrite, global_indices=None):
    """Generate encoding images for one dataset/split/encoding combo."""
    out_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images",
                           encoding_name, split)
    os.makedirs(out_dir, exist_ok=True)

    generated = 0
    skipped = 0

    for i, ts in enumerate(X):
        idx = global_indices[i] if global_indices is not None else i
        path = os.path.join(out_dir, f"sample_{idx}.png")

        if os.path.exists(path) and not overwrite:
            skipped += 1
            continue

        try:
            img = get_encoding(encoding_name, ts, image_size)
            save_encoding_image(img, path)
            generated += 1
        except Exception as e:
            print(f"    ERROR: {dataset_name}/{encoding_name}/sample_{idx}: {e}")

    return generated, skipped


def generate_rgb_stack_images(dataset_name, X, y, split, preset_name,
                               image_size, overwrite, global_indices=None):
    """Generate RGB-stacked images."""
    out_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images",
                           f"rgb_{preset_name}", split)
    os.makedirs(out_dir, exist_ok=True)

    generated = 0
    skipped = 0

    for i, ts in enumerate(X):
        idx = global_indices[i] if global_indices is not None else i
        path = os.path.join(out_dir, f"sample_{idx}.png")

        if os.path.exists(path) and not overwrite:
            skipped += 1
            continue

        try:
            rgb = get_rgb_stack(preset_name, ts, image_size)
            from PIL import Image
            Image.fromarray(rgb, mode='RGB').save(path)
            generated += 1
        except Exception as e:
            print(f"    ERROR: {dataset_name}/rgb_{preset_name}/sample_{idx}: {e}")

    return generated, skipped


def generate_colormap_images(dataset_name, X, y, split, base_encoding,
                              colormap, image_size, overwrite,
                              global_indices=None):
    """Generate colormapped versions of grayscale encodings."""
    out_dir = os.path.join(CHART_IMAGE_ROOT, f"{dataset_name}_images",
                           f"{base_encoding}_{colormap}", split)
    os.makedirs(out_dir, exist_ok=True)

    generated = 0
    skipped = 0

    for i, ts in enumerate(X):
        idx = global_indices[i] if global_indices is not None else i
        path = os.path.join(out_dir, f"sample_{idx}.png")

        if os.path.exists(path) and not overwrite:
            skipped += 1
            continue

        try:
            gray = get_encoding(base_encoding, ts, image_size)
            rgb = apply_colormap(gray, colormap)
            from PIL import Image
            Image.fromarray(rgb, mode='RGB').save(path)
            generated += 1
        except Exception as e:
            print(f"    ERROR: {dataset_name}/{base_encoding}_{colormap}/sample_{idx}: {e}")

    return generated, skipped


def process_dataset(dataset_name, args):
    """Process one dataset: generate all requested encodings."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    X_train, y_train, X_test, y_test = load_dataset(dataset_name)
    if X_train is None:
        print(f"  SKIP: dataset not found")
        return 0, 0

    n_train = len(X_train)
    n_test = len(X_test)
    print(f"  Samples: {n_train} train + {n_test} test = {n_train + n_test} total")
    print(f"  TS length: {X_train[0].shape[-1] if hasattr(X_train[0], 'shape') else len(X_train[0])}")

    # Global indices: train uses 0..n_train-1, test uses n_train..n_train+n_test-1
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_train + n_test))

    total_gen = 0
    total_skip = 0

    # --- Mathematical encodings ---
    for enc_name in args.encodings:
        t0 = time.time()
        g1, s1 = generate_encoding_images(
            dataset_name, X_train, y_train, "train", enc_name,
            args.image_size, args.overwrite, train_indices
        )
        g2, s2 = generate_encoding_images(
            dataset_name, X_test, y_test, "test", enc_name,
            args.image_size, args.overwrite, test_indices
        )
        elapsed = time.time() - t0
        gen = g1 + g2
        skip = s1 + s2
        total_gen += gen
        total_skip += skip
        if gen > 0:
            print(f"  {enc_name:20s}: {gen:>5} new, {skip:>5} exist ({elapsed:.1f}s)")

    # --- RGB stacks ---
    if args.rgb_presets:
        for preset in args.rgb_presets:
            t0 = time.time()
            g1, s1 = generate_rgb_stack_images(
                dataset_name, X_train, y_train, "train", preset,
                args.image_size, args.overwrite, train_indices
            )
            g2, s2 = generate_rgb_stack_images(
                dataset_name, X_test, y_test, "test", preset,
                args.image_size, args.overwrite, test_indices
            )
            elapsed = time.time() - t0
            gen = g1 + g2
            total_gen += gen
            total_skip += s1 + s2
            if gen > 0:
                print(f"  rgb_{preset:16s}: {gen:>5} new ({elapsed:.1f}s)")

    # --- Colormaps ---
    if args.colormaps:
        colormap_bases = ['gasf', 'gadf', 'rp_grayscale']
        for base_enc in colormap_bases:
            if base_enc not in args.encodings:
                continue
            for cmap in args.colormaps:
                t0 = time.time()
                g1, s1 = generate_colormap_images(
                    dataset_name, X_train, y_train, "train",
                    base_enc, cmap, args.image_size, args.overwrite, train_indices
                )
                g2, s2 = generate_colormap_images(
                    dataset_name, X_test, y_test, "test",
                    base_enc, cmap, args.image_size, args.overwrite, test_indices
                )
                elapsed = time.time() - t0
                gen = g1 + g2
                total_gen += gen
                total_skip += s1 + s2
                if gen > 0:
                    print(f"  {base_enc}_{cmap:10s}: {gen:>5} new ({elapsed:.1f}s)")

    print(f"  TOTAL: {total_gen} generated, {total_skip} skipped")
    return total_gen, total_skip


def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate ALL image encodings for VTBench experiments"
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Specific datasets (default: core 5)")
    parser.add_argument("--extended", action="store_true",
                        help="Use extended 15 datasets")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Use all 32 UCR datasets")
    parser.add_argument("--encodings", nargs="+", default=DEFAULT_ENCODINGS,
                        help=f"Encodings to generate (default: all). "
                             f"Options: {DEFAULT_ENCODINGS}")
    parser.add_argument("--rgb-presets", nargs="+", default=DEFAULT_RGB_PRESETS,
                        help=f"RGB stack presets. Options: {DEFAULT_RGB_PRESETS}")
    parser.add_argument("--colormaps", nargs="+", default=DEFAULT_COLORMAPS,
                        help="Colormaps to apply (default: viridis, plasma)")
    parser.add_argument("--no-rgb", action="store_true",
                        help="Skip RGB stacking")
    parser.add_argument("--no-colormaps", action="store_true",
                        help="Skip colormap variants")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Output image size (default: 128)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing images")

    args = parser.parse_args()

    if args.no_rgb:
        args.rgb_presets = []
    if args.no_colormaps:
        args.colormaps = []

    if args.all_datasets:
        datasets = ALL_DATASETS
    elif args.extended:
        datasets = EXTENDED_DATASETS
    elif args.datasets:
        datasets = args.datasets
    else:
        datasets = CORE_DATASETS

    print("=" * 60)
    print("VTBench — Image Encoding Pre-Generation")
    print("=" * 60)
    print(f"Datasets:   {len(datasets)}")
    print(f"Encodings:  {args.encodings}")
    print(f"RGB stacks: {args.rgb_presets}")
    print(f"Colormaps:  {args.colormaps}")
    print(f"Image size: {args.image_size}")
    print(f"Output:     {CHART_IMAGE_ROOT}")
    print(f"Overwrite:  {args.overwrite}")

    # Estimate total images
    est_per_dataset = 7000  # rough estimate
    n_enc = len(args.encodings)
    n_rgb = len(args.rgb_presets) if args.rgb_presets else 0
    n_cmap = len(args.colormaps) * 3 if args.colormaps else 0  # 3 base encodings
    n_variants = n_enc + n_rgb + n_cmap
    print(f"Variants:   {n_variants} per dataset ({n_enc} enc + {n_rgb} rgb + {n_cmap} cmap)")
    print()

    start_time = time.time()
    grand_gen = 0
    grand_skip = 0

    for i, ds in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}]", end="")
        gen, skip = process_dataset(ds, args)
        grand_gen += gen
        grand_skip += skip

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"DONE.")
    print(f"  Generated: {grand_gen:,}")
    print(f"  Skipped:   {grand_skip:,}")
    print(f"  Time:      {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Images at: {CHART_IMAGE_ROOT}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
