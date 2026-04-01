#!/usr/bin/env python
"""
Quick fix: scan for and regenerate corrupt images.
Specifically targets colormap variants (gasf_viridis, gadf_plasma,
rp_grayscale_inferno) which are known to have corrupt files.

Usage:
    python scripts/fix_corrupt_images.py
    python scripts/fix_corrupt_images.py --dry-run   # report only
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image
from vtbench.data.loader import read_ucr
from vtbench.data.ts_image_encodings import (
    get_encoding, apply_colormap, save_encoding_image,
    ENCODING_REGISTRY,
)

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")

# Encodings that use colormaps (most likely to be corrupt)
COLORMAP_ENCODINGS = {
    "gasf_viridis":         ("gasf", "viridis"),
    "gadf_plasma":          ("gadf", "plasma"),
    "rp_grayscale_inferno": ("rp_grayscale", "inferno"),
}

# Also check all ENCODING_REGISTRY entries
ALL_ENCODINGS = list(ENCODING_REGISTRY.keys()) + list(COLORMAP_ENCODINGS.keys())

DATASETS = [
    "CBF", "ECG200", "ECG5000", "FordA", "GunPoint",
    "ItalyPowerDemand", "SyntheticControl", "Trace",
    "TwoLeadECG", "Wafer", "Yoga", "Lightning2",
    "Herring", "Wine", "Beef", "CricketZ", "Earthquakes",
    "FaceAll", "Fish", "FordB", "SwedishLeaf",
    "ToeSegmentation1", "TwoPatterns", "WordSynonyms",
    "ChlorineConcentration", "ArrowHead", "BeetleFly",
    "Computers", "Crop",
]


def verify_image(path):
    """Return True if image is valid, False if corrupt/missing."""
    if not os.path.exists(path):
        return False
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--encodings", nargs="+", default=None,
                        help="Specific encodings to check (default: colormaps + all)")
    args = parser.parse_args()

    encodings_to_check = args.encodings or list(COLORMAP_ENCODINGS.keys())
    total_corrupt = 0
    total_fixed = 0

    for ds in DATASETS:
        # Load dataset
        X_train, y_train, X_test, y_test = None, None, None, None
        for ext in [".tsv", ".ts"]:
            tp = f"UCRArchive_2018/{ds}/{ds}_TRAIN{ext}"
            ep = f"UCRArchive_2018/{ds}/{ds}_TEST{ext}"
            if os.path.exists(tp):
                X_train, y_train = read_ucr(tp)
                X_test, y_test = read_ucr(ep)
                break
        if X_train is None:
            continue

        n_train = len(X_train)

        for enc_name in encodings_to_check:
            for split, X, offset in [("train", X_train, 0),
                                     ("test", X_test, n_train)]:
                enc_dir = os.path.join(CHART_IMAGE_ROOT,
                                       f"{ds}_images", enc_name, split)
                if not os.path.isdir(enc_dir):
                    continue

                for i, ts in enumerate(X):
                    gidx = offset + i
                    path = os.path.join(enc_dir, f"sample_{gidx}.png")
                    if verify_image(path):
                        continue

                    total_corrupt += 1
                    if args.dry_run:
                        print(f"  CORRUPT: {path}")
                        continue

                    # Regenerate
                    try:
                        if enc_name in COLORMAP_ENCODINGS:
                            base, cmap = COLORMAP_ENCODINGS[enc_name]
                            gray = get_encoding(base, ts, 128)
                            arr = apply_colormap(gray, cmap)
                        else:
                            arr = get_encoding(enc_name, ts, 128)
                        save_encoding_image(arr, path)
                        total_fixed += 1
                    except Exception as e:
                        print(f"  ERROR regenerating {path}: {e}")

        if total_corrupt > 0:
            print(f"  {ds}: found {total_corrupt} corrupt so far")

    print(f"\nTotal corrupt: {total_corrupt}")
    if not args.dry_run:
        print(f"Total fixed: {total_fixed}")
    else:
        print("(dry-run mode — no files modified)")


if __name__ == "__main__":
    main()
