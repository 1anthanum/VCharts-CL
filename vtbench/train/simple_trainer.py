"""
Standalone training loop for image classification.

This is used by encoding-based experiments (6A, 7A, 8A, 8B) that bypass
the chart_generator pipeline and load pre-generated images directly.

The original trainer.py handles the full chart pipeline (single_modal_chart,
two_branch, multi_modal_chart); this module handles the simpler case of
"given a model and dataloaders, train and return accuracy."
"""

import time

import torch
import torch.nn as nn


def train_and_evaluate(model, train_loader, test_loader, config, device=None):
    """Train a model and return test accuracy.

    Args:
        model: nn.Module to train
        train_loader: DataLoader for training
        test_loader: DataLoader for evaluation
        config: dict with keys 'epochs', 'learning_rate' (optional)
        device: torch device (auto-detected if None)

    Returns:
        float: Best validation/test accuracy achieved
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = config.get("epochs", 100)
    lr = config.get("learning_rate", 0.001)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0
    patience = config.get("patience", 10)

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = correct / total if total > 0 else 0

        # --- Validate ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = val_correct / val_total if val_total > 0 else 0

        scheduler.step(val_loss)

        print(
            f"  [Epoch {epoch + 1}] "
            f"Train Acc: {train_acc:.2%}, "
            f"Val Acc: {val_acc:.2%}, "
            f"Val Loss: {val_loss:.4f}"
        )

        # --- Early stopping ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch + 1}.")
                break

    return best_val_acc


def train_and_profile(model, train_loader, test_loader, config, device=None):
    """Train a model and return accuracy + profiling metrics.

    Returns:
        dict with keys:
            accuracy, train_time_s, epochs_trained,
            peak_gpu_mb, inference_ms_mean, inference_ms_std, n_params
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = config.get("epochs", 100)
    lr = config.get("learning_rate", 0.001)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    n_params = sum(p.numel() for p in model.parameters())

    # Reset GPU memory tracking
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    best_val_acc = 0
    patience_counter = 0
    patience = config.get("patience", 10)
    epochs_trained = 0

    t0 = time.time()

    for epoch in range(epochs):
        epochs_trained = epoch + 1
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
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

    train_time = time.time() - t0

    # Peak GPU memory
    peak_gpu_mb = 0
    if device == "cuda":
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Inference latency
    inference_ms = measure_inference_latency(model, train_loader, device)

    return {
        "accuracy": best_val_acc,
        "train_time_s": train_time,
        "epochs_trained": epochs_trained,
        "peak_gpu_mb": peak_gpu_mb,
        "inference_ms_mean": inference_ms["mean"],
        "inference_ms_std": inference_ms["std"],
        "n_params": n_params,
    }


def measure_inference_latency(model, loader, device, n_warmup=10, n_runs=100):
    """Measure per-sample inference latency in milliseconds."""
    model.eval()

    # Get a sample batch
    sample_batch = next(iter(loader))[0][:1].to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample_batch)

    # Timed runs
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample_batch)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    import numpy as np
    times = np.array(times)
    return {"mean": float(times.mean()), "std": float(times.std())}
