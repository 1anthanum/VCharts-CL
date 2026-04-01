"""
Time-series level augmentations (applied BEFORE chart generation).
=================================================================
These augment the raw numerical time series, then new chart images
are generated from the augmented data. This is fundamentally different
from image-domain augmentations (blur, noise, JPEG) used in experiments
4 and 5A.

Supported augmentations:
    - jitter:       Add small Gaussian noise to values
    - scaling:      Multiply by a random scalar near 1.0
    - time_warp:    Smooth non-linear time warping
    - window_slice: Random contiguous subsequence (with resize back)
    - window_warp:  Speed up/slow down a random window
    - magnitude_warp: Smooth magnitude scaling via cubic spline
    - permutation:  Randomly permute segments
"""

import numpy as np
from scipy.interpolate import CubicSpline


def ts_jitter(ts, sigma=0.03):
    """Add Gaussian noise to time series values."""
    return ts + np.random.normal(0, sigma, size=ts.shape)


def ts_scaling(ts, sigma=0.1):
    """Scale the entire time series by a random factor near 1.0."""
    factor = np.random.normal(1.0, sigma)
    return ts * factor


def ts_time_warp(ts, sigma=0.2, num_knots=4):
    """Smooth non-linear time warping using cubic spline."""
    orig_steps = np.arange(len(ts))
    random_warps = np.random.normal(1.0, sigma, size=num_knots + 2)
    warp_steps = np.linspace(0, len(ts) - 1, num=num_knots + 2)
    warper = CubicSpline(warp_steps, warp_steps * random_warps)(orig_steps)
    warper = np.clip(warper, 0, len(ts) - 1)

    # Interpolate
    scale = (len(ts) - 1) / (warper[-1] - warper[0] + 1e-8)
    warped_indices = (warper - warper[0]) * scale
    warped_indices = np.clip(warped_indices, 0, len(ts) - 1)

    from numpy import interp
    return interp(orig_steps, warped_indices, ts)


def ts_window_slice(ts, reduce_ratio=0.9):
    """Take a random contiguous window and resize back to original length."""
    target_len = max(2, int(len(ts) * reduce_ratio))
    start = np.random.randint(0, len(ts) - target_len + 1)
    sliced = ts[start:start + target_len]
    # Resize back using linear interpolation
    return np.interp(
        np.linspace(0, target_len - 1, num=len(ts)),
        np.arange(target_len),
        sliced,
    )


def ts_window_warp(ts, window_ratio=0.1, scales=(0.5, 2.0)):
    """Speed up or slow down a random window of the time series."""
    warp_size = max(2, int(len(ts) * window_ratio))
    start = np.random.randint(0, len(ts) - warp_size + 1)
    scale = np.random.uniform(scales[0], scales[1])

    window = ts[start:start + warp_size]
    new_size = max(2, int(warp_size * scale))
    warped_window = np.interp(
        np.linspace(0, warp_size - 1, num=new_size),
        np.arange(warp_size),
        window,
    )

    # Reconstruct: before + warped_window + after
    before = ts[:start]
    after = ts[start + warp_size:]
    combined = np.concatenate([before, warped_window, after])

    # Resize back to original length
    return np.interp(
        np.linspace(0, len(combined) - 1, num=len(ts)),
        np.arange(len(combined)),
        combined,
    )


def ts_magnitude_warp(ts, sigma=0.2, num_knots=4):
    """Smooth magnitude scaling via cubic spline."""
    orig_steps = np.arange(len(ts))
    knot_xs = np.linspace(0, len(ts) - 1, num=num_knots + 2)
    knot_ys = np.random.normal(1.0, sigma, size=num_knots + 2)
    spline = CubicSpline(knot_xs, knot_ys)(orig_steps)
    return ts * spline


def ts_permutation(ts, max_segments=5):
    """Randomly permute segments of the time series."""
    num_segments = np.random.randint(2, max_segments + 1)
    segment_len = len(ts) // num_segments
    if segment_len < 2:
        return ts.copy()

    segments = []
    for i in range(num_segments):
        start = i * segment_len
        end = start + segment_len if i < num_segments - 1 else len(ts)
        segments.append(ts[start:end])

    np.random.shuffle(segments)
    return np.concatenate(segments)


# ====================================================================
# Registry
# ====================================================================

TS_AUGMENTATION_REGISTRY = {
    'jitter': ts_jitter,
    'scaling': ts_scaling,
    'time_warp': ts_time_warp,
    'window_slice': ts_window_slice,
    'window_warp': ts_window_warp,
    'magnitude_warp': ts_magnitude_warp,
    'permutation': ts_permutation,
}


def apply_ts_augmentation(ts, aug_name, **kwargs):
    """Apply a named time-series augmentation.

    Parameters
    ----------
    ts : np.ndarray (T,)
        1D time series.
    aug_name : str
        Augmentation name from TS_AUGMENTATION_REGISTRY.
    **kwargs
        Augmentation-specific parameters.

    Returns
    -------
    np.ndarray (T,) — augmented time series
    """
    fn = TS_AUGMENTATION_REGISTRY.get(aug_name)
    if fn is None:
        raise ValueError(f"Unknown TS augmentation: {aug_name}. "
                         f"Available: {list(TS_AUGMENTATION_REGISTRY.keys())}")
    return fn(ts, **kwargs)


def augment_dataset(X, y, num_augmented=1, aug_names=None, seed=42):
    """Generate augmented copies of a dataset.

    Parameters
    ----------
    X : np.ndarray (N, T) — original time series
    y : np.ndarray (N,) — labels
    num_augmented : int
        Number of augmented copies per original sample.
    aug_names : list of str or None
        Which augmentations to randomly apply. If None, uses all.
    seed : int

    Returns
    -------
    X_aug, y_aug — augmented arrays (includes originals)
    """
    rng = np.random.RandomState(seed)
    if aug_names is None:
        aug_names = list(TS_AUGMENTATION_REGISTRY.keys())

    X_all = [X]
    y_all = [y]

    for _ in range(num_augmented):
        X_new = np.zeros_like(X)
        for i in range(len(X)):
            aug_name = rng.choice(aug_names)
            X_new[i] = apply_ts_augmentation(X[i], aug_name)
        X_all.append(X_new)
        y_all.append(y.copy())

    return np.vstack(X_all), np.concatenate(y_all)
