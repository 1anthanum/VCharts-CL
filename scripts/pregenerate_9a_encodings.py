#!/usr/bin/env python
"""
Pre-generate encoding images needed by Experiment 9A.
=====================================================
Generates: gadf, cwt, rp_grayscale, wavelet_scattering, signature, persistence
for 5 small datasets × train/test splits.

Run BEFORE experiment_9a_advanced.py:
    python scripts/pregenerate_9a_encodings.py

This is CPU-only and takes ~10-20 minutes.
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vtbench.data.loader import read_ucr
from vtbench.data.ts_image_encodings import get_encoding

CHART_IMAGE_ROOT = os.environ.get("CHART_IMAGE_ROOT", "chart_images")
DATASET_ROOT = "UCRArchive_2018"
IMAGE_SIZE = 128

DATASETS = ["SyntheticControl", "GunPoint", "CBF", "Trace", "ECG5000"]

# All encodings needed by 9A
ENCODINGS = ["gadf", "cwt", "rp_grayscale", "wavelet_scattering", "signature", "persistence"]


def generate_encoding_images(dataset_name, encoding_name, image_size=128):
    """Generate encoding images for one dataset × one encoding."""
    dataset_dir = os.path.join(DATASET_ROOT, dataset_name)

    # Read data
    try:
        X_train, y_train, X_test, y_test = read_ucr(dataset_dir, ext="tsv")
    except Exception as e:
        print(f"  ERROR reading {dataset_name}: {e}")
        return 0

    total = 0
    for split, X in [("train", X_train), ("test", X_test)]:
        out_dir = os.path.join(
            CHART_IMAGE_ROOT, f"{dataset_name}_images", encoding_name, split
        )
        os.makedirs(out_dir, exist_ok=True)

        for idx in range(len(X)):
            out_path = os.path.join(out_dir, f"sample_{idx}.png")
            if os.path.exists(out_path):
                total += 1
                continue

            ts = X[idx]
            try:
                img_array = get_encoding(encoding_name, ts, image_size)
                if img_array.ndim == 2:
                    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='L').convert('RGB')
                elif img_array.ndim == 3 and img_array.shape[2] == 3:
                    img = Image.fromarray((img_array * 255).astype(np.uint8), mode='RGB')
                else:
                    img = Image.fromarray((img_array * 255).astype(np.uint8))
                    img = img.convert('RGB')
                img = img.resize((image_size, image_size), Image.LANCZOS)
                img.save(out_path)
                total += 1
            except Exception as e:
                print(f"  ERROR {dataset_name}/{encoding_name}/{split}/sample_{idx}: {e}")
                # Create grey placeholder
                img = Image.new("RGB", (image_size, image_size), (128, 128, 128))
                img.save(out_path)
                total += 1

    return total


def main():
    print("=" * 60)
    print("Pre-generating encoding images for Experiment 9A")
    print(f"CHART_IMAGE_ROOT: {CHART_IMAGE_ROOT}")
    print(f"Datasets: {DATASETS}")
    print(f"Encodings: {ENCODINGS}")
    print("=" * 60)

    t0 = time.time()
    grand_total = 0

    for ds in DATASETS:
        for enc in ENCODINGS:
            t1 = time.time()
            count = generate_encoding_images(ds, enc, IMAGE_SIZE)
            elapsed = time.time() - t1
            print(f"  {ds}/{enc}: {count} images ({elapsed:.1f}s)")
            grand_total += count

    elapsed_total = time.time() - t0
    print(f"\nDone! {grand_total} images in {elapsed_total:.1f}s")


if __name__ == "__main__":
    main()
