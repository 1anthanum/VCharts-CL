"""
Training utilities: Mixup, Label Smoothing, Cosine Annealing, Class Weights.
============================================================================
These are drop-in helpers used by experiment scripts that need enhanced
training beyond the default trainer.py loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter


# ====================================================================
# Mixup
# ====================================================================

def mixup_data(x, y, alpha=0.2):
    """Apply mixup to a batch of images and labels.

    Parameters
    ----------
    x : Tensor (B, C, H, W)
    y : Tensor (B,) integer labels
    alpha : float
        Beta distribution parameter. 0 = no mixup.

    Returns
    -------
    mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ====================================================================
# CutMix
# ====================================================================

def cutmix_data(x, y, alpha=1.0):
    """Apply CutMix to a batch of images.

    Parameters
    ----------
    x : Tensor (B, C, H, W)
    y : Tensor (B,) integer labels
    alpha : float
        Beta distribution parameter.

    Returns
    -------
    mixed_x, y_a, y_b, lam
    """
    if alpha <= 0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = max(0, cx - cut_w // 2)
    y1 = max(0, cy - cut_h // 2)
    x2 = min(W, cx + cut_w // 2)
    y2 = min(H, cy + cut_h // 2)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda based on actual area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    return mixed_x, y, y[index], lam


# ====================================================================
# Label Smoothing Cross Entropy
# ====================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    Parameters
    ----------
    smoothing : float
        Label smoothing factor (0.0 = standard CE, 0.1 = typical).
    """

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# ====================================================================
# Class-weighted Cross Entropy
# ====================================================================

def get_class_weights(labels, device='cpu'):
    """Compute inverse-frequency class weights for imbalanced datasets.

    Parameters
    ----------
    labels : list or array of integer labels
    device : str

    Returns
    -------
    torch.Tensor of shape (num_classes,)
    """
    counter = Counter(labels)
    total = sum(counter.values())
    num_classes = max(counter.keys()) + 1
    weights = torch.zeros(num_classes)
    for cls, count in counter.items():
        weights[cls] = total / (num_classes * count)
    return weights.to(device)


# ====================================================================
# Enhanced training loop
# ====================================================================

def train_with_enhancements(model, train_loader, val_loader, config, device='cuda'):
    """Training loop with optional mixup, label smoothing, cosine LR.

    Config keys (under 'training'):
        - learning_rate: float
        - epochs: int
        - weight_decay: float (default 0.01)
        - mixup_alpha: float (default 0.0 = disabled)
        - cutmix_alpha: float (default 0.0 = disabled)
        - label_smoothing: float (default 0.0 = disabled)
        - scheduler: 'plateau' | 'cosine' (default 'plateau')
        - class_weights: bool (default False)

    Returns
    -------
    model (trained, on device)
    """
    import torch.optim as optim

    training_cfg = config.get('training', {})
    lr = training_cfg.get('learning_rate', 0.001)
    epochs = training_cfg.get('epochs', 100)
    weight_decay = training_cfg.get('weight_decay', 0.01)
    mixup_alpha = training_cfg.get('mixup_alpha', 0.0)
    cutmix_alpha = training_cfg.get('cutmix_alpha', 0.0)
    label_smoothing = training_cfg.get('label_smoothing', 0.0)
    scheduler_type = training_cfg.get('scheduler', 'plateau')
    use_class_weights = training_cfg.get('class_weights', False)

    model = model.to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5
        )

    # Loss
    if label_smoothing > 0:
        criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    elif use_class_weights:
        labels = [int(label) for _, label in train_loader.dataset]
        weights = get_class_weights(labels, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Early stopping
    patience = training_cfg.get('patience', 10)
    trigger_times = 0
    best_val_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Mixup or CutMix (randomly choose one if both enabled)
            use_mix = False
            if mixup_alpha > 0 and cutmix_alpha > 0:
                if np.random.random() < 0.5:
                    images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
                else:
                    images, targets_a, targets_b, lam = cutmix_data(images, labels, cutmix_alpha)
                use_mix = True
            elif mixup_alpha > 0:
                images, targets_a, targets_b, lam = mixup_data(images, labels, mixup_alpha)
                use_mix = True
            elif cutmix_alpha > 0:
                images, targets_a, targets_b, lam = cutmix_data(images, labels, cutmix_alpha)
                use_mix = True

            optimizer.zero_grad()
            outputs = model(images)

            if use_mix:
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)  # always standard CE for val
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")

        # Scheduler step
        if scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping.")
                break

    return model
