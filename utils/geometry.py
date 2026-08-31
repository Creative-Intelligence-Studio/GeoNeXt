"""Convert one RGB/depth/normal prediction into a vertex-colored mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fov_x(cls, width: int, height: int, fov_x_degrees: float = 60.0) -> "Camera":
        fx = width / (2.0 * np.tan(np.deg2rad(fov_x_degrees) / 2.0))
        return cls(float(fx), float(fx), (width - 1) / 2.0, (height - 1) / 2.0)

    @classmethod
    def from_normalized_matrix(cls, matrix: np.ndarray, width: int, height: int) -> "Camera":
        k = np.asarray(matrix, dtype=np.float32).squeeze()
        return cls(float(k[0, 0] * width), float(k[1, 1] * height),
                   float(k[0, 2] * width), float(k[1, 2] * height))


def disparity_to_relative_depth(
    disparity: np.ndarray,
    eps: float = 1e-6,
    min_disparity: float = 0.1,
) -> np.ndarray:
    """Convert normalized disparity to a bounded relative-depth map.

    Very small predicted disparities are unreliable after inversion and can
    otherwise create extreme mesh spikes. Clamp them to a conservative floor
    before converting to relative depth.
    """
    d = np.asarray(disparity, dtype=np.float32).squeeze()
    finite = np.isfinite(d)
    if not finite.any():
        raise ValueError("Disparity contains no finite values")
    lo, hi = np.percentile(d[finite], [1.0, 99.0])
    d01 = np.clip((d - lo) / max(float(hi - lo), eps), 0.0, 1.0)
    depth = 1.0 / np.maximum(d01, max(float(min_disparity), eps))
    median = np.median(depth[finite])
    return depth / max(float(median), eps)


def _resize_rgb(rgb: Union[np.ndarray, Image.Image], width: int, height: int) -> np.ndarray:
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(np.asarray(rgb, dtype=np.uint8))
    return np.asarray(image.convert("RGB").resize((width, height), Image.Resampling.BILINEAR))


def _normalize_normals(normal: Optional[np.ndarray], height: int, width: int) -> Optional[np.ndarray]:
    if normal is None:
        return None
    n = np.asarray(normal, dtype=np.float32).squeeze()
    if n.shape == (3, height, width):
        n = np.moveaxis(n, 0, -1)
    if n.shape != (height, width, 3):
        channels = [Image.fromarray(n[..., i], mode="F").resize((width, height), Image.Resampling.BILINEAR)
                    for i in range(3)]
        n = np.stack([np.asarray(x) for x in channels], axis=-1)
    return n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-8)


def build_surface(
    rgb: Union[np.ndarray, Image.Image],
    depth: np.ndarray,
    camera: Camera,
    normal: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    stride: int = 1,
    edge_threshold: float = 0.03,
    normal_angle_threshold: float = 70.0,
    normal_bypass_depth_threshold: float = 0.015,
) -> Dict[str, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    height, width = depth.shape
    rgb = _resize_rgb(rgb, width, height)
    normal = _normalize_normals(normal, height, width)
    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid &= np.asarray(mask).squeeze().astype(bool)

    ys, xs = np.arange(0, height, stride), np.arange(0, width, stride)
    z = depth[np.ix_(ys, xs)]
    keep = valid[np.ix_(ys, xs)]
    uu, vv = np.meshgrid(xs, ys)
    xyz = np.stack(((uu - camera.cx) * z / camera.fx,
                    -(vv - camera.cy) * z / camera.fy, -z), axis=-1).astype(np.float32)
    colors = rgb[np.ix_(ys, xs)].astype(np.uint8)
    normals = normal[np.ix_(ys, xs)].astype(np.float32) if normal is not None else None
    if normals is not None:
        normals[..., 1:] *= -1
        # OpenGL camera space looks along -Z, so front-facing surface normals
        # should point roughly towards +Z. Checkpoint conventions may differ by
        # one global sign; correct that sign without changing local geometry.
        finite_normals = np.isfinite(normals).all(axis=-1)
        if finite_normals.any() and np.median(normals[..., 2][finite_normals]) < 0:
            normals *= -1

    rows, cols = z.shape
    index = np.arange(rows * cols, dtype=np.int32).reshape(rows, cols)
    faces = np.concatenate((
        np.stack((index[:-1, :-1], index[:-1, 1:], index[1:, :-1]), -1).reshape(-1, 3),
        np.stack((index[:-1, 1:], index[1:, 1:], index[1:, :-1]), -1).reshape(-1, 3),
    ))
    flat_z, flat_keep = z.reshape(-1), keep.reshape(-1)
    tri_z = flat_z[faces]
    relative_jump = (tri_z.max(1) - tri_z.min(1)) / np.maximum(tri_z.min(1), 1e-6)
    face_keep = flat_keep[faces].all(1) & (relative_jump < edge_threshold)
    if normals is not None and normal_angle_threshold > 0:
        flat_normals = normals.reshape(-1, 3)
        triangle_normals = flat_normals[faces]
        pair_cosine = np.stack((
            (triangle_normals[:, 0] * triangle_normals[:, 1]).sum(1),
            (triangle_normals[:, 1] * triangle_normals[:, 2]).sum(1),
            (triangle_normals[:, 2] * triangle_normals[:, 0]).sum(1),
        ), axis=1)
        normal_ok = (np.isfinite(triangle_normals).all((1, 2))
                     & (pair_cosine.min(1) > math.cos(math.radians(normal_angle_threshold))))
        if normal_bypass_depth_threshold > 0:
            normal_ok |= relative_jump < normal_bypass_depth_threshold
        face_keep &= normal_ok
    faces = faces[face_keep]

    used = np.zeros(rows * cols, dtype=bool)
    used[faces.ravel()] = True
    remap = np.full(rows * cols, -1, dtype=np.int32)
    remap[used] = np.arange(used.sum(), dtype=np.int32)
    result = {
        "vertices": xyz.reshape(-1, 3)[used], "faces": remap[faces],
        "colors": colors.reshape(-1, 3)[used],
    }
    if normals is not None:
        result["normals"] = normals.reshape(-1, 3)[used]
    return result


def export_geometry(output_dir: Union[str, Path], rgb: Union[np.ndarray, Image.Image], depth: np.ndarray,
                    camera: Camera, normal: Optional[np.ndarray] = None, mask: Optional[np.ndarray] = None,
                    exports: Iterable[str] = ("mesh",), stride: int = 1,
                    edge_threshold: float = 0.04,
                    normal_angle_threshold: float = 70.0,
                    normal_bypass_depth_threshold: float = 0.015) -> Dict[str, Path]:
    """Export a triangle PLY with RGB vertex colors."""
    import trimesh
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = set(exports)
    unsupported = exports - {"mesh"}
    if unsupported:
        raise ValueError("Unsupported geometry export(s): %s" % ", ".join(sorted(unsupported)))
    surface = build_surface(
        rgb, depth, camera, normal, mask, stride, edge_threshold,
        normal_angle_threshold, normal_bypass_depth_threshold)
    kwargs = {"vertices": surface["vertices"], "vertex_colors": surface["colors"], "process": False}
    if "normals" in surface:
        kwargs["vertex_normals"] = surface["normals"]
    written = {}  # type: Dict[str, Path]
    if "mesh" in exports:
        path = output_dir / "mesh.ply"
        trimesh.Trimesh(faces=surface["faces"], **kwargs).export(path)
        written["mesh"] = path
    return written
