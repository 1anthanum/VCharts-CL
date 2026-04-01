#!/usr/bin/env python
"""
Pre-flight Check: Audit existing images and report what needs generation.
=========================================================================
Scans CHART_IMAGE_ROOT for all datasets x encoding types, reports:
  - Which images already exist (with counts)
  - Which are MISSING and need generation
  - Generates missing images automatically (unless --dry-run)

Usage:
    python scripts/preflight_check.py --dry-run          # audit only
    python scripts/preflight_check.py                    # audit + generate
    python scripts/preflight_check.py --datasets GunPoint ECG5000  # subset
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")

# Auto-discover all datasets in UCRArchive_2018/
_UCR_ROOT = os.environ.get("UCR_ROOT", "UCRArchive_2018")
if os.path.isdir(_UCR_ROOT):
    ALL_DATASETS = sorted([
        d for d in os.listdir(_UCR_ROOT)
        if os.path.isdir(os.path.join(_UCR_ROOT, d))
        and not d.startswith(".")
        and not d.startswith("Missing_value")
        and any("TRAIN" in f for f in os.listdir(os.path.join(_UCR_ROOT, d)))
    ])
else:
    ALL_DATASETS = [
        "SyntheticControl", "GunPoint", "CBF", "Trace", "ECG5000", "FordA",
        "Wafer", "TwoPatterns", "BeetleFly", "TwoLeadECG", "Coffee",
        "Lightning2", "ECG200", "Computers", "Yoga", "ArrowHead", "Beef",
        "Wine", "ItalyPowerDemand", "SwedishLeaf", "Fish", "Adiac",
    ]

# Chart types to check
CHART_TYPES = {
    "line": "line_charts_color_with_label",
    "area": "area_charts_color_with_label",
    "bar": "bar_charts_border_color_with_label",
    "scatter": "scatter_charts_plain_color_with_label",
}

CHART_TYPES_NOLABEL = {
    "line_nolabel": "line_charts_color_without_label",
    "area_nolabel": "area_charts_color_without_label",
}

# Encoding types to check
MATH_ENCODINGS = [
    "gasf", "gadf", "mtf", "rp", "rp_grayscale", "cwt", "stft",
]

RGB_PRESETS = [
    "rgb_gasf_gadf_mtf", "rgb_rp_cwt_stft",
]

EXTENDED_ENCODINGS = [
    "phase_scatter", "phase_trajectory", "phase_tau_auto",
    "phase_multi_tau", "gasf_diff", "gasf_cumsum", "rp_diff",
]

EXTENDED_RGB = [
    "rgb_phase_gasf_rp", "rgb_phase_cwt_mtf", "rgb_gasf_gasfdiff_gadf",
]

COLORMAP_ENCODINGS = [
    "gasf_viridis", "gadf_plasma", "rp_grayscale_inferno",
]


def count_images(dirpath):
    """Count PNG files across train/test subdirs."""
    if not os.path.isdir(dirpath):
        return 0
    total = 0
    for split in ["train", "test"]:
        sp = os.path.join(dirpath, split)
        if os.path.isdir(sp):
            total += len([f for f in os.listdir(sp) if f.endswith(".png")])
    # Also count flat PNGs (no split subdirs)
    if total == 0:
        total = len([f for f in os.listdir(dirpath) if f.endswith(".png")])
    return total


def get_expected_count(dataset_name):
    """Get expected sample count for a dataset."""
    for ext in ["tsv", "ts"]:
        train_path = f"UCRArchive_2018/{dataset_name}/{dataset_name}_TRAIN.{ext}"
        test_path = f"UCRArchive_2018/{dataset_name}/{dataset_name}_TEST.{ext}"
        if os.path.exists(train_path):
            with open(train_path) as f:
                n_train = len(f.readlines())
            with open(test_path) as f:
                n_test = len(f.readlines())
            return n_train + n_test
    return 0


def audit_images(datasets):
    """Audit all image directories and return status."""
    results = {}  # {dataset: {encoding: (exists, count, expected)}}

    for ds in datasets:
        expected = get_expected_count(ds)
        base = os.path.join(CHART_IMAGE_ROOT, f"{ds}_images")
        ds_results = {}

        # Chart types (with label)
        for name, dirname in CHART_TYPES.items():
            dirpath = os.path.join(base, dirname)
            count = count_images(dirpath)
            ds_results[f"chart:{name}"] = (count > 0, count, expected)

        # Chart types (without label)
        for name, dirname in CHART_TYPES_NOLABEL.items():
            dirpath = os.path.join(base, dirname)
            count = count_images(dirpath)
            ds_results[f"chart:{name}"] = (count > 0, count, expected)

        # Math encodings
        for enc in MATH_ENCODINGS:
            dirpath = os.path.join(base, enc)
            count = count_images(dirpath)
            ds_results[f"math:{enc}"] = (count > 0, count, expected)

        # RGB presets
        for enc in RGB_PRESETS:
            dirpath = os.path.join(base, enc)
            count = count_images(dirpath)
            ds_results[f"rgb:{enc}"] = (count > 0, count, expected)

        # Extended encodings
        for enc in EXTENDED_ENCODINGS:
            dirpath = os.path.join(base, enc)
            count = count_images(dirpath)
            ds_results[f"ext:{enc}"] = (count > 0, count, expected)

        # Extended RGB
        for enc in EXTENDED_RGB:
            dirpath = os.path.join(base, enc)
            count = count_images(dirpath)
            ds_results[f"ext_rgb:{enc}"] = (count > 0, count, expected)

        # Colormaps
        for enc in COLORMAP_ENCODINGS:
            dirpath = os.path.join(base, enc)
            count = count_images(dirpath)
            ds_results[f"cmap:{enc}"] = (count > 0, count, expected)

        results[ds] = ds_results

    return results


def print_audit(results):
    """Print audit summary."""
    total_exist = 0
    total_missing = 0
    missing_work = []  # (dataset, encoding_type, encoding_name)

    for ds, encodings in sorted(results.items()):
        existing = sum(1 for v in encodings.values() if v[0])
        missing = sum(1 for v in encodings.values() if not v[0])
        total_exist += existing
        total_missing += missing

        if missing > 0:
            print(f"\n{ds}: {existing} OK, {missing} MISSING")
            for enc_key, (exists, count, expected) in sorted(encodings.items()):
                if not exists:
                    enc_type, enc_name = enc_key.split(":", 1)
                    print(f"  MISSING: {enc_key}")
                    missing_work.append((ds, enc_type, enc_name))
        else:
            total_imgs = sum(v[1] for v in encodings.values())
            print(f"{ds}: ALL {existing} encodings present ({total_imgs} images)")

    print(f"\n{'='*60}")
    print(f"TOTAL: {total_exist} encoding/dataset combos exist, "
          f"{total_missing} missing")

    return missing_work


def generate_missing_charts(ds, chart_type, label_mode="with_label"):
    """Generate missing chart images for a dataset."""
    from vtbench.data.loader import read_ucr
    from vtbench.data.chart_generator import TimeSeriesImageDataset
    from torch.utils.data import DataLoader

    for ext in ["tsv", "ts"]:
        train_path = f"UCRArchive_2018/{ds}/{ds}_TRAIN.{ext}"
        test_path = f"UCRArchive_2018/{ds}/{ds}_TEST.{ext}"
        if os.path.exists(train_path):
            break
    else:
        print(f"  Dataset {ds} not found, skipping")
        return

    X_train, y_train = read_ucr(train_path)
    X_test, y_test = read_ucr(test_path)

    n_train = len(X_train)
    n_test = len(X_test)

    for split, X, y, offset in [
        ("train", X_train, y_train, 0),
        ("test", X_test, y_test, n_train),
    ]:
        indices = list(range(offset, offset + len(X)))
        ts_data = list(X)
        labels = list(y)

        ct = chart_type.replace("_nolabel", "")
        lm = "without_label" if "nolabel" in chart_type else label_mode

        # Images are generated during __init__ when generate_images=True
        TimeSeriesImageDataset(
            time_series_data=ts_data,
            labels=labels,
            dataset_name=ds,
            split=split,
            chart_type=ct,
            color_mode="color",
            label_mode=lm,
            generate_images=True,
            global_indices=indices,
        )


def generate_missing_encodings(ds, enc_name):
    """Generate missing encoding images for a dataset."""
    from vtbench.data.loader import read_ucr
    from vtbench.data.ts_image_encodings import (
        ENCODING_REGISTRY, RGB_STACK_PRESETS,
        get_encoding, get_rgb_stack, save_encoding_image,
        apply_colormap,
    )

    for ext in ["tsv", "ts"]:
        train_path = f"UCRArchive_2018/{ds}/{ds}_TRAIN.{ext}"
        test_path = f"UCRArchive_2018/{ds}/{ds}_TEST.{ext}"
        if os.path.exists(train_path):
            break
    else:
        print(f"  Dataset {ds} not found, skipping")
        return

    X_train, y_train = read_ucr(train_path)
    X_test, y_test = read_ucr(test_path)

    n_train = len(X_train)
    image_size = 128

    is_rgb = enc_name.startswith("rgb_")
    is_colormap = any(enc_name.endswith(f"_{cm}") for cm in
                      ["viridis", "plasma", "inferno", "magma", "jet"])

    for split, X, offset in [
        ("train", X_train, 0),
        ("test", X_test, n_train),
    ]:
        out_dir = os.path.join(CHART_IMAGE_ROOT, f"{ds}_images", enc_name, split)
        os.makedirs(out_dir, exist_ok=True)

        for i, ts in enumerate(X):
            gidx = offset + i
            out_path = os.path.join(out_dir, f"sample_{gidx}.png")
            if os.path.exists(out_path):
                # Validate existing image isn't corrupt
                try:
                    from PIL import Image as PILImage
                    PILImage.open(out_path).verify()
                    continue
                except Exception:
                    os.remove(out_path)  # Remove corrupt file

            try:
                if is_rgb:
                    preset_name = enc_name.replace("rgb_", "")
                    img = get_rgb_stack(preset_name, ts, image_size)
                elif is_colormap:
                    parts = enc_name.rsplit("_", 1)
                    base_enc, cmap = parts[0], parts[1]
                    gray = get_encoding(base_enc, ts, image_size)
                    img = apply_colormap(gray, cmap)
                else:
                    img = get_encoding(enc_name, ts, image_size)

                save_encoding_image(img, out_path)
            except Exception as e:
                if i == 0:
                    print(f"  Error on first sample: {e}")
                    break


def main():
    parser = argparse.ArgumentParser(description="Pre-flight image audit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit only, do not generate missing images")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets (default: all)")
    args = parser.parse_args()

    datasets = args.datasets or ALL_DATASETS

    print(f"CHART_IMAGE_ROOT: {CHART_IMAGE_ROOT}")
    print(f"Auditing {len(datasets)} datasets...")
    print(f"{'='*60}")

    results = audit_images(datasets)
    missing_work = print_audit(results)

    if not missing_work:
        print("\nAll images present. Ready to run experiments!")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] {len(missing_work)} encoding/dataset combos need generation.")
        print("Run without --dry-run to generate them.")
        return

    print(f"\n{'='*60}")
    print(f"Generating {len(missing_work)} missing image sets...")
    print(f"{'='*60}")

    t0 = time.time()
    for i, (ds, enc_type, enc_name) in enumerate(missing_work, 1):
        print(f"\n[{i}/{len(missing_work)}] {ds} / {enc_name} ({enc_type})")

        try:
            if enc_type == "chart":
                generate_missing_charts(ds, enc_name)
            else:
                generate_missing_encodings(ds, enc_name)
            print(f"  Done.")
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - t0
    print(f"\nImage generation complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # Re-audit
    print(f"\n{'='*60}")
    print("Re-auditing...")
    results2 = audit_images(datasets)
    missing2 = print_audit(results2)
    if not missing2:
        print("\nAll images now present. Ready to run experiments!")


if __name__ == "__main__":
    main()
