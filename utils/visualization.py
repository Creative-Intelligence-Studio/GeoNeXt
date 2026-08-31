"""Shared prediction visualization and output helpers."""

from pathlib import Path

import numpy as np
from PIL import Image


# ColorBrewer Spectral (reversed by default for disparity-style predictions).
_SPECTRAL = np.asarray(
    [
        [158, 1, 66], [213, 62, 79], [244, 109, 67], [253, 174, 97],
        [254, 224, 139], [255, 255, 191], [230, 245, 152], [171, 221, 164],
        [102, 194, 165], [50, 136, 189], [94, 79, 162],
    ],
    dtype=np.float32,
)


def depth_to_vis(depth, reverse_color=True):
    """Convert a 2-D depth/disparity array to an RGB visualization."""
    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("depth must have shape [H, W]")
    finite = np.isfinite(values)
    normalized = np.zeros_like(values)
    if finite.any():
        low = float(values[finite].min())
        high = float(values[finite].max())
        if high > low:
            normalized[finite] = (values[finite] - low) / (high - low)
    if reverse_color:
        normalized = 1.0 - normalized
    position = normalized * (len(_SPECTRAL) - 1)
    lower = np.floor(position).astype(np.intp)
    upper = np.minimum(lower + 1, len(_SPECTRAL) - 1)
    weight = (position - lower)[..., None]
    rgb = _SPECTRAL[lower] * (1.0 - weight) + _SPECTRAL[upper] * weight
    rgb[~finite] = 0
    return Image.fromarray(np.rint(rgb).astype(np.uint8), mode="RGB")


def normal_to_vis(normal):
    """Convert normal vectors in [-1, 1] to an RGB visualization."""
    values = np.asarray(normal, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("normal must have shape [H, W, 3]")
    rgb = np.rint((np.clip(values, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_prediction_outputs(output_root, stem, rgb, depth, normal):
    """Save raw predictions and their visualizations using the release layout."""
    root = Path(output_root)
    directories = {
        name: root / name
        for name in ("depth_raw", "depth_vis", "normal_raw", "normal_vis", "rgb_recon")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    # Both public backends store normalized disparity in [0, 1].
    depth = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
    normal = np.asarray(normal, dtype=np.float32)
    np.save(directories["depth_raw"] / (stem + ".npy"), depth)
    np.save(directories["normal_raw"] / (stem + ".npy"), normal)
    depth_to_vis(depth, reverse_color=True).save(directories["depth_vis"] / (stem + ".png"))
    normal_to_vis(normal).save(directories["normal_vis"] / (stem + ".png"))
    rgb.convert("RGB").save(directories["rgb_recon"] / (stem + ".png"))
