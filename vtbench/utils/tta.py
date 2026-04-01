"""
Test-Time Augmentation (TTA)
=============================
Apply multiple augmented views to each test image at inference time,
then average the predictions. Improves accuracy by 1-3% with zero
additional training cost.

Usage:
    from vtbench.utils.tta import tta_predict

    # Instead of: outputs = model(images)
    outputs = tta_predict(model, images, n_augments=5)
"""

import torch
import torch.nn.functional as F
from torchvision import transforms


# Default TTA transforms: mild geometric + color perturbations
def get_tta_transforms(image_size=128):
    """Generate a list of TTA transform functions.

    Each function takes a (B, C, H, W) tensor and returns augmented version.
    """
    tta_list = [
        # Original (identity)
        lambda x: x,
        # Horizontal flip
        lambda x: torch.flip(x, dims=[3]),
        # Small rotation (via affine grid, ~5 degrees)
        lambda x: _rotate_batch(x, 5.0),
        lambda x: _rotate_batch(x, -5.0),
        # Slight scale (center crop 90% + resize back)
        lambda x: _center_crop_scale(x, 0.9),
        # Slight brightness shift
        lambda x: torch.clamp(x * 1.05, -3, 3),
        lambda x: torch.clamp(x * 0.95, -3, 3),
    ]
    return tta_list


def _rotate_batch(x, angle_deg):
    """Rotate batch by angle degrees using affine grid."""
    B, C, H, W = x.shape
    angle_rad = angle_deg * 3.14159265 / 180.0
    cos_a = torch.cos(torch.tensor(angle_rad))
    sin_a = torch.sin(torch.tensor(angle_rad))

    # 2x3 affine matrix for rotation
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0]
    ], dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1, -1)

    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, mode='bilinear', padding_mode='border')


def _center_crop_scale(x, ratio):
    """Center crop to ratio of original size, then resize back."""
    B, C, H, W = x.shape
    crop_h = int(H * ratio)
    crop_w = int(W * ratio)
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2
    cropped = x[:, :, top:top + crop_h, left:left + crop_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


@torch.no_grad()
def tta_predict(model, images, n_augments=5):
    """
    Make predictions with Test-Time Augmentation.

    Parameters
    ----------
    model : nn.Module
        Trained model in eval mode.
    images : Tensor (B, C, H, W)
        Input batch.
    n_augments : int
        Number of augmented views (including original). Max 7.

    Returns
    -------
    Tensor (B, num_classes)
        Averaged logits across augmented views.
    """
    tta_fns = get_tta_transforms()[:n_augments]

    all_logits = []
    for fn in tta_fns:
        aug_images = fn(images)
        logits = model(aug_images)
        all_logits.append(logits)

    # Average logits (more stable than averaging probabilities)
    return torch.stack(all_logits).mean(dim=0)
