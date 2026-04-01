"""
Deterministic image-domain augmentations for the Augmentation Robustness
Experiment.

All transforms operate on PIL Images and are fully deterministic (no
random state) so results are reproducible across runs.

Supported augmentation types:
    - gaussian_blur:   PIL GaussianBlur with configurable sigma.
    - crop_horizontal: Deterministic horizontal crop + resize back.
    - translate:       Fixed horizontal pixel shift (pad + crop).
    - stripe_mask:     Vertical stripe masking at specified positions.
    - gaussian_noise:  Additive Gaussian noise (deterministic seed).
    - jpeg_compress:   JPEG compression at given quality level.
"""

import io
import numpy as np
from PIL import Image, ImageFilter


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_augmentation(img, aug_type, params):
    """Apply a single deterministic augmentation to a PIL Image.

    Parameters
    ----------
    img : PIL.Image
        Input image (RGB).
    aug_type : str
        One of the supported augmentation names.
    params : dict
        Augmentation-specific parameters.

    Returns
    -------
    PIL.Image
        Augmented image (same size as input, RGB).
    """
    img = img.convert("RGB")
    fn = _REGISTRY.get(aug_type)
    if fn is None:
        raise ValueError(
            f"Unsupported augmentation type: {aug_type}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return fn(img, params)


def augmentation_label(aug_type, params):
    """Return a human-readable label for an augmentation + params pair."""
    if aug_type == "gaussian_blur":
        return f"blur_s{params.get('sigma', 1.0)}"
    if aug_type == "crop_horizontal":
        pct = params.get("crop_fraction", 0.05)
        return f"crop_{int(pct * 100)}pct"
    if aug_type == "translate":
        px = params.get("pixels", 0)
        return f"translate_{px}px"
    if aug_type == "stripe_mask":
        n = params.get("num_stripes", 3)
        w = params.get("stripe_width", 5)
        return f"stripe_{n}x{w}"
    if aug_type == "gaussian_noise":
        s = params.get("sigma", 10)
        return f"noise_s{s}"
    if aug_type == "jpeg_compress":
        q = params.get("quality", 50)
        return f"jpeg_q{q}"
    return f"{aug_type}"


# ---------------------------------------------------------------------------
# Individual augmentation implementations
# ---------------------------------------------------------------------------

def _gaussian_blur(img, params):
    """Apply Gaussian blur.

    params:
        sigma (float): Gaussian sigma, default 1.0.
    """
    sigma = float(params.get("sigma", 1.0))
    # PIL GaussianBlur uses 'radius' (kernel half-size); we approximate
    # a reasonable kernel radius from sigma.
    radius = max(1, int(round(sigma * 2)))
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _crop_horizontal(img, params):
    """Deterministic horizontal crop then resize back to original size.

    params:
        crop_fraction (float): Fraction to remove, e.g. 0.05 for 5%.
            Half is removed from each side.
    """
    frac = float(params.get("crop_fraction", 0.05))
    w, h = img.size
    margin = int(round(w * frac / 2))
    if margin < 1:
        return img.copy()
    cropped = img.crop((margin, 0, w - margin, h))
    return cropped.resize((w, h), resample=Image.BILINEAR)


def _translate(img, params):
    """Fixed horizontal translation (positive = shift right, pad left).

    params:
        pixels (int): Number of pixels to shift. Negative = shift left.
    """
    px = int(params.get("pixels", 0))
    if px == 0:
        return img.copy()
    w, h = img.size
    arr = np.asarray(img)
    result = np.full_like(arr, 255)  # white fill
    if px > 0:
        px = min(px, w)
        result[:, px:, :] = arr[:, :w - px, :]
    else:
        px = min(-px, w)
        result[:, :w - px, :] = arr[:, px:, :]
    return Image.fromarray(result)


def _stripe_mask(img, params):
    """Mask narrow vertical stripes at evenly-spaced positions.

    params:
        num_stripes (int): Number of stripes, default 3.
        stripe_width (int): Width per stripe in pixels, default 5.
        fill (tuple): RGB fill value, default (255, 255, 255) white.
    """
    n = int(params.get("num_stripes", 3))
    sw = int(params.get("stripe_width", 5))
    fill = tuple(params.get("fill", (255, 255, 255)))
    w, h = img.size
    arr = np.asarray(img).copy()
    # Evenly spaced stripe centres
    if n < 1:
        return img.copy()
    positions = np.linspace(0, w - 1, n + 2)[1:-1].astype(int)
    for cx in positions:
        x0 = max(0, cx - sw // 2)
        x1 = min(w, x0 + sw)
        arr[:, x0:x1, :] = fill
    return Image.fromarray(arr)


def _gaussian_noise(img, params):
    """Add deterministic Gaussian noise.

    params:
        sigma (float): Noise standard deviation (pixel scale 0-255), default 10.
        seed (int): Fixed RNG seed for reproducibility, default 0.
    """
    sigma = float(params.get("sigma", 10))
    seed = int(params.get("seed", 0))
    rng = np.random.RandomState(seed)
    arr = np.asarray(img).astype(np.float32)
    noise = rng.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _jpeg_compress(img, params):
    """Simulate JPEG compression artifacts.

    params:
        quality (int): JPEG quality 1-95, default 50.
    """
    quality = int(params.get("quality", 50))
    quality = max(1, min(95, quality))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY = {
    "gaussian_blur": _gaussian_blur,
    "crop_horizontal": _crop_horizontal,
    "translate": _translate,
    "stripe_mask": _stripe_mask,
    "gaussian_noise": _gaussian_noise,
    "jpeg_compress": _jpeg_compress,
}
