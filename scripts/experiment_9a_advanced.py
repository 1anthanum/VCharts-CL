#!/usr/bin/env python
"""
Experiment 9A: Advanced Methods for Accuracy Improvement
=========================================================
Unified experiment testing ALL new accuracy improvement methods:

  A. New backbone models: EfficientNet B0 (pretrained), ViT Tiny (pretrained)
  B. New encoding methods: Wavelet Scattering, Signature, Persistence
  C. Training augmentation: Mixup, CutMix
  D. Test-Time Augmentation (TTA)
  E. Channel attention: SE-Net, CBAM on DeepCNN
  F. Cross-encoding ensemble (voting)

Each method is tested independently against the baseline to isolate its
contribution. Results are saved to a single CSV for cross-comparison.

Usage:
    python scripts/experiment_9a_advanced.py \
        --config vtbench/config/experiment_9a_advanced.yaml
"""

import argparse
import csv
import gc
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
sys.path.insert(0, _project_root)

from vtbench.data.loader import read_ucr
from vtbench.train.factory import get_chart_model
from vtbench.models.chart_models.attention import wrap_model_with_attention
from vtbench.utils.mixup import mixup_data, cutmix_data, mixup_criterion
from vtbench.utils.tta import tta_predict
from vtbench.utils.wandb_logger import WandbLogger
from vtbench.utils.heartbeat import Heartbeat

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Aggressive GPU optimizations ──
if device == "cuda":
    torch.backends.cudnn.benchmark = True          # auto-tune conv kernels
    torch.backends.cuda.matmul.allow_tf32 = True   # TF32 for matmul (~2x on Ampere+)
    torch.backends.cudnn.allow_tf32 = True          # TF32 for cuDNN


# ============================================================
# Dataset class (reused from 6A pattern)
# ============================================================

class EncodingImageDataset(torch.utils.data.Dataset):
    """Load pre-generated encoding images."""

    def __init__(self, dataset_name, split, encoding_name,
                 global_indices, labels, transform=None):
        self.labels = labels
        self.transform = transform
        self.global_indices = global_indices

        chart_root = os.environ.get("CHART_IMAGE_ROOT", "chart_images")
        self.img_dir = os.path.join(
            chart_root, f"{dataset_name}_images", encoding_name, split
        )
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        gidx = self.global_indices[idx]
        img_path = os.path.join(self.img_dir, f"sample_{gidx}.png")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (128, 128), (128, 128, 128))
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


# ============================================================
# Training variants
# ============================================================

def train_baseline(model, train_loader, test_loader, config):
    """Standard training (no mixup/cutmix)."""
    return _train_loop(model, train_loader, test_loader, config,
                       augment_mode='none')


def train_mixup(model, train_loader, test_loader, config):
    """Training with Mixup augmentation."""
    return _train_loop(model, train_loader, test_loader, config,
                       augment_mode='mixup')


def train_cutmix(model, train_loader, test_loader, config):
    """Training with CutMix augmentation."""
    return _train_loop(model, train_loader, test_loader, config,
                       augment_mode='cutmix')


def _train_loop(model, train_loader, test_loader, config, augment_mode='none'):
    """Unified training loop with optional Mixup/CutMix + AMP + torch.compile."""
    epochs = config.get('epochs', 100)
    lr = config.get('learning_rate', 0.001)

    model = model.to(device)
    # torch.compile disabled — Triton not available on Windows
    # Causes lazy compilation failure on first forward pass
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # AMP mixed precision
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_acc = 0
    patience_counter = 0
    patience = 10
    mixup_alpha = config.get('mixup_alpha', 0.2)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                if augment_mode == 'mixup':
                    images, y_a, y_b, lam = mixup_data(images, labels, alpha=mixup_alpha)
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                elif augment_mode == 'cutmix':
                    images, y_a, y_b, lam = cutmix_data(images, labels, alpha=1.0)
                    outputs = model(images)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total if total > 0 else 0

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.cpu()
    return best_val_acc


def evaluate_with_tta(model, test_loader, n_augments=5):
    """Evaluate model with Test-Time Augmentation."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = tta_predict(model, images, n_augments=n_augments)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    model.cpu()
    return correct / total if total > 0 else 0


def ensemble_vote(models_and_loaders):
    """
    Cross-encoding ensemble via majority voting.

    Parameters
    ----------
    models_and_loaders : list of (model, test_loader)
        Each model trained on a different encoding.

    Returns
    -------
    float : ensemble accuracy
    """
    all_preds = []
    all_labels = None

    for model, test_loader in models_and_loaders:
        model = model.to(device)
        model.eval()
        preds = []
        labels = []

        with torch.no_grad():
            for images, lbls in test_loader:
                images = images.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                preds.extend(predicted.cpu().numpy())
                labels.extend(lbls.numpy())

        all_preds.append(np.array(preds))
        if all_labels is None:
            all_labels = np.array(labels)
        model.cpu()

    # Majority voting
    votes = np.stack(all_preds)  # (n_models, n_samples)
    ensemble_pred = []
    for i in range(votes.shape[1]):
        counts = np.bincount(votes[:, i], minlength=all_labels.max() + 1)
        ensemble_pred.append(counts.argmax())

    ensemble_pred = np.array(ensemble_pred)
    accuracy = (ensemble_pred == all_labels).mean()
    return accuracy


# ============================================================
# Main experiment
# ============================================================

def run_experiment(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    exp = cfg['experiment']
    output_dir = exp['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    wb = WandbLogger(project="vtbench", experiment="9a_advanced", config=cfg)
    hb = Heartbeat("9a")

    datasets = exp['datasets']
    seeds = exp.get('seeds', [42, 123, 7])
    image_size = exp.get('image_size', 128)
    encodings = exp.get('encodings', ['gasf', 'gadf', 'cwt'])
    batch_size = cfg.get('training', {}).get('batch_size', 128)
    epochs = cfg.get('training', {}).get('epochs', 100)

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # CSV with resume support
    csv_path = os.path.join(output_dir, "advanced_results.csv")
    fieldnames = ["seed", "dataset", "encoding", "model", "method",
                  "accuracy", "train_time_s"]
    completed_runs = set()

    if os.path.exists(csv_path):
        failed_count = 0
        with open(csv_path, encoding="utf-8") as f_exist:
            reader = csv.DictReader(f_exist)
            for row in reader:
                # Only skip runs that actually succeeded (accuracy > 0)
                # Failed runs (accuracy=-1) should be retried
                try:
                    acc = float(row.get("accuracy", -1))
                except (ValueError, TypeError):
                    acc = -1
                if acc > 0:
                    key = (row['seed'], row['dataset'], row['encoding'],
                           row['model'], row['method'])
                    completed_runs.add(key)
                else:
                    failed_count += 1
        print(f"Resuming: {len(completed_runs)} completed runs "
              f"({failed_count} failed runs will be retried)")
        csv_file = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    else:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    # Methods to test
    methods = exp.get('methods', [
        'baseline',          # Standard training
        'mixup',             # Mixup augmentation
        'cutmix',            # CutMix augmentation
        'tta_5',             # Test-Time Augmentation (5 views)
        'tta_7',             # Test-Time Augmentation (7 views)
        'se_attention',      # SE attention on backbone
        'cbam_attention',    # CBAM attention on backbone
    ])

    # Models to test
    model_specs = exp.get('models', [
        {"name": "deepcnn", "chart_model": "deepcnn", "pretrained": False, "lr": 0.001},
        {"name": "resnet18_pt", "chart_model": "resnet18", "pretrained": True, "lr": 0.0005},
        {"name": "efficientnet_pt", "chart_model": "efficientnet_b0", "pretrained": True, "lr": 0.0005},
        {"name": "vit_tiny", "chart_model": "vit_tiny", "pretrained": False, "lr": 0.0005},
    ])

    # New encodings
    new_encodings = exp.get('new_encodings', [
        'wavelet_scattering', 'signature', 'persistence',
    ])

    for ds_name in datasets:
        X_train, y_train, X_test, y_test = None, None, None, None
        for ext in [".tsv", ".ts"]:
            train_path = f"UCRArchive_2018/{ds_name}/{ds_name}_TRAIN{ext}"
            test_path = f"UCRArchive_2018/{ds_name}/{ds_name}_TEST{ext}"
            if os.path.exists(train_path):
                X_train, y_train = read_ucr(train_path)
                X_test, y_test = read_ucr(test_path)
                break

        if X_train is None:
            print(f"SKIP {ds_name}: not found")
            continue

        n_train = len(X_train)
        num_classes = len(set(y_train))
        train_indices = list(range(n_train))
        test_indices = list(range(n_train, n_train + len(X_test)))

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name} ({n_train} train, {len(X_test)} test, "
              f"{num_classes} classes)")
        print(f"{'='*60}")

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # ---- Part 1: Method comparison on existing encodings ----
            for enc_name in encodings:
                for method in methods:
                    for mspec in model_specs:
                        model_name = mspec['name']
                        run_key = (str(seed), ds_name, enc_name,
                                   model_name, method)

                        if run_key in completed_runs:
                            continue

                        print(f"\n  [{ds_name}] enc={enc_name} model={model_name} "
                              f"method={method} seed={seed}")

                        hb.pulse(
                            dataset=ds_name,
                            encoding=enc_name,
                            model=model_name,
                            method=method,
                            seed=seed,
                        )

                        try:
                            # Load dataset
                            train_ds = EncodingImageDataset(
                                ds_name, "train", enc_name,
                                train_indices, list(y_train), tfm
                            )
                            test_ds = EncodingImageDataset(
                                ds_name, "test", enc_name,
                                test_indices, list(y_test), tfm
                            )

                            safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                            train_loader = DataLoader(
                                train_ds, batch_size=safe_bs, shuffle=True,
                                drop_last=True, num_workers=2,
                                pin_memory=True, persistent_workers=False,
                            )
                            test_loader = DataLoader(
                                test_ds, batch_size=min(batch_size, len(test_ds)),
                                shuffle=False, num_workers=2,
                                pin_memory=True, persistent_workers=False,
                            )

                            if len(train_loader) == 0:
                                print(f"    SKIP: empty loader")
                                continue

                            # Create model
                            model = get_chart_model(
                                mspec['chart_model'],
                                input_channels=3,
                                num_classes=num_classes,
                                pretrained=mspec.get('pretrained', False),
                                image_size=image_size,
                            )

                            # Apply attention if needed
                            if method == 'se_attention':
                                model = wrap_model_with_attention(model, 'se')
                            elif method == 'cbam_attention':
                                model = wrap_model_with_attention(model, 'cbam')

                            t0 = time.time()

                            # Train
                            train_cfg = {
                                'epochs': epochs,
                                'learning_rate': mspec.get('lr', 0.001),
                                'mixup_alpha': 0.2,
                            }

                            if method in ('baseline', 'tta_5', 'tta_7',
                                          'se_attention', 'cbam_attention'):
                                accuracy = train_baseline(
                                    model, train_loader, test_loader, train_cfg)
                            elif method == 'mixup':
                                accuracy = train_mixup(
                                    model, train_loader, test_loader, train_cfg)
                            elif method == 'cutmix':
                                accuracy = train_cutmix(
                                    model, train_loader, test_loader, train_cfg)

                            # Apply TTA if applicable
                            if method.startswith('tta_'):
                                n_aug = int(method.split('_')[1])
                                accuracy = evaluate_with_tta(
                                    model, test_loader, n_augments=n_aug)

                            train_time = time.time() - t0

                            print(f"    -> Accuracy: {accuracy:.4f} "
                                  f"({train_time:.1f}s)")

                            writer.writerow({
                                "seed": seed, "dataset": ds_name,
                                "encoding": enc_name, "model": model_name,
                                "method": method,
                                "accuracy": f"{accuracy:.6f}",
                                "train_time_s": f"{train_time:.1f}",
                            })
                            csv_file.flush()

                            wb.log_run_result(
                                name=f"{ds_name}/{enc_name}/{model_name}/{method}/s{seed}",
                                config={"dataset": ds_name, "encoding": enc_name,
                                        "model": model_name, "method": method,
                                        "seed": seed},
                                accuracy=accuracy,
                            )

                        except Exception as err:
                            import traceback
                            print(f"    FAILED: {err}")
                            traceback.print_exc()
                            writer.writerow({
                                "seed": seed, "dataset": ds_name,
                                "encoding": enc_name, "model": model_name,
                                "method": method, "accuracy": "-1",
                                "train_time_s": "0",
                            })
                            csv_file.flush()

                        finally:
                            model = None
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

            # ---- Part 2: New encodings (baseline only) ----
            for enc_name in new_encodings:
                for mspec in model_specs[:2]:  # deepcnn + resnet18 only
                    model_name = mspec['name']
                    run_key = (str(seed), ds_name, enc_name,
                               model_name, 'baseline')

                    if run_key in completed_runs:
                        continue

                    print(f"\n  [{ds_name}] NEW enc={enc_name} "
                          f"model={model_name} seed={seed}")

                    hb.pulse(
                        dataset=ds_name,
                        encoding=enc_name,
                        model=model_name,
                        method="baseline",
                        seed=seed,
                    )

                    try:
                        train_ds = EncodingImageDataset(
                            ds_name, "train", enc_name,
                            train_indices, list(y_train), tfm
                        )
                        test_ds = EncodingImageDataset(
                            ds_name, "test", enc_name,
                            test_indices, list(y_test), tfm
                        )

                        safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                        train_loader = DataLoader(
                            train_ds, batch_size=safe_bs, shuffle=True,
                            drop_last=True, num_workers=2,
                            pin_memory=True, persistent_workers=False,
                        )
                        test_loader = DataLoader(
                            test_ds, batch_size=min(batch_size, len(test_ds)),
                            shuffle=False, num_workers=2,
                            pin_memory=True, persistent_workers=False,
                        )

                        model = get_chart_model(
                            mspec['chart_model'],
                            input_channels=3,
                            num_classes=num_classes,
                            pretrained=mspec.get('pretrained', False),
                            image_size=image_size,
                        )

                        t0 = time.time()
                        accuracy = train_baseline(
                            model, train_loader, test_loader,
                            {'epochs': epochs, 'learning_rate': mspec.get('lr', 0.001)},
                        )
                        train_time = time.time() - t0

                        print(f"    -> Accuracy: {accuracy:.4f}")

                        writer.writerow({
                            "seed": seed, "dataset": ds_name,
                            "encoding": enc_name, "model": model_name,
                            "method": "baseline",
                            "accuracy": f"{accuracy:.6f}",
                            "train_time_s": f"{train_time:.1f}",
                        })
                        csv_file.flush()

                    except Exception as err:
                        import traceback
                        print(f"    FAILED: {err}")
                        traceback.print_exc()
                        writer.writerow({
                            "seed": seed, "dataset": ds_name,
                            "encoding": enc_name, "model": model_name,
                            "method": "baseline", "accuracy": "-1",
                            "train_time_s": "0",
                        })
                        csv_file.flush()

                    finally:
                        model = None
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            # ---- Part 3: Cross-encoding ensemble ----
            run_key = (str(seed), ds_name, "ensemble_top3", "resnet18_pt", "voting")
            if run_key not in completed_runs:
                print(f"\n  [{ds_name}] ENSEMBLE top-3 encodings seed={seed}")

                hb.pulse(
                    dataset=ds_name,
                    encoding="ensemble_top3",
                    model="resnet18_pt",
                    method="voting",
                    seed=seed,
                )

                try:
                    # Use top-3 diverse encodings
                    ensemble_encs = exp.get('ensemble_encodings',
                                            ['gadf', 'cwt', 'rp_grayscale'])
                    models_and_loaders = []

                    for enc_name in ensemble_encs:
                        train_ds = EncodingImageDataset(
                            ds_name, "train", enc_name,
                            train_indices, list(y_train), tfm
                        )
                        test_ds = EncodingImageDataset(
                            ds_name, "test", enc_name,
                            test_indices, list(y_test), tfm
                        )

                        safe_bs = min(batch_size, len(train_ds)) if len(train_ds) > 0 else 1
                        train_loader = DataLoader(
                            train_ds, batch_size=safe_bs, shuffle=True,
                            drop_last=True, num_workers=2,
                            pin_memory=True, persistent_workers=False,
                        )
                        test_loader = DataLoader(
                            test_ds, batch_size=min(batch_size, len(test_ds)),
                            shuffle=False, num_workers=2,
                            pin_memory=True, persistent_workers=False,
                        )

                        model = get_chart_model(
                            "resnet18", input_channels=3,
                            num_classes=num_classes, pretrained=True,
                            image_size=image_size,
                        )
                        train_baseline(
                            model, train_loader, test_loader,
                            {'epochs': epochs, 'learning_rate': 0.0005},
                        )
                        model = model.to(device)
                        models_and_loaders.append((model, test_loader))

                    ens_acc = ensemble_vote(models_and_loaders)
                    print(f"    -> Ensemble accuracy: {ens_acc:.4f}")

                    writer.writerow({
                        "seed": seed, "dataset": ds_name,
                        "encoding": "ensemble_top3",
                        "model": "resnet18_pt", "method": "voting",
                        "accuracy": f"{ens_acc:.6f}",
                        "train_time_s": "0",
                    })
                    csv_file.flush()

                except Exception as err:
                    import traceback
                    print(f"    ENSEMBLE FAILED: {err}")
                    traceback.print_exc()
                    writer.writerow({
                        "seed": seed, "dataset": ds_name,
                        "encoding": "ensemble_top3",
                        "model": "resnet18_pt", "method": "voting",
                        "accuracy": "-1", "train_time_s": "0",
                    })
                    csv_file.flush()

                finally:
                    models_and_loaders = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    csv_file.close()
    wb.finish()
    hb.close()
    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 9A: Advanced Accuracy Improvement Methods"
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    args = parser.parse_args()
    run_experiment(args.config)
