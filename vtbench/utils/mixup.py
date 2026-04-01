"""
Mixup and CutMix Training Augmentations
========================================
Data-level regularization techniques that create virtual training examples
by mixing pairs of images and their labels.

- Mixup:  x' = λ·x_i + (1-λ)·x_j,  y' = λ·y_i + (1-λ)·y_j
- CutMix: paste a random patch from x_j onto x_i, mix labels by area ratio

References:
  Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization"
  Yun et al. (2019) "CutMix: Regularization Strategy to Train Strong Classifiers"
"""

import numpy as np
import torch


def mixup_data(x, y, alpha=0.2):
    """
    Apply mixup to a batch of data.

    Parameters
    ----------
    x : Tensor (B, C, H, W)
        Input images.
    y : Tensor (B,)
        Labels (integer class indices).
    alpha : float
        Beta distribution parameter. Higher = more mixing.

    Returns
    -------
    mixed_x : Tensor
    y_a, y_b : Tensor
        Original and shuffled labels.
    lam : float
        Mixing coefficient.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    """
    Apply CutMix to a batch of data.

    Parameters
    ----------
    x : Tensor (B, C, H, W)
    y : Tensor (B,)
    alpha : float
        Beta distribution parameter for area ratio.

    Returns
    -------
    mixed_x : Tensor
    y_a, y_b : Tensor
    lam : float
        Effective mixing ratio (by area).
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    _, _, H, W = x.shape

    # Sample bounding box
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    # Center of box
    cy = np.random.randint(H)
    cx = np.random.randint(W)

    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)

    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    # Adjust lambda to actual area ratio
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixed loss: lam * L(pred, y_a) + (1-lam) * L(pred, y_b)."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
