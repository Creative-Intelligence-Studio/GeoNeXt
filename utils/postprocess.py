"""Command-line geometry exporter for GeoNeXt prediction folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.geometry import Camera, disparity_to_relative_depth, export_geometry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True, help="NPY depth or disparity")
    parser.add_argument("--normal", help="NPY normal in [-1,1]")
    parser.add_argument("--mask", help="Optional NPY mask")
    parser.add_argument("--intrinsics", help="Optional normalized 3x3 NPY intrinsics")
    parser.add_argument("--fov-x", type=float, default=60.0)
    parser.add_argument("--depth-mode", choices=["depth", "disparity"], default="disparity")
    parser.add_argument("--export", nargs="+", default=["mesh"], choices=["mesh"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--normal-angle-threshold", type=float, default=70.0)
    parser.add_argument("--normal-bypass-depth-threshold", type=float, default=0.015)
    args = parser.parse_args()

    rgb = Image.open(args.rgb).convert("RGB")
    depth = np.load(args.depth).squeeze().astype(np.float32)
    if args.depth_mode == "disparity":
        depth = disparity_to_relative_depth(depth)
    normal = np.load(args.normal) if args.normal else None
    mask = np.load(args.mask) if args.mask else None
    h, w = depth.shape
    camera = (Camera.from_normalized_matrix(np.load(args.intrinsics), w, h)
              if args.intrinsics else Camera.from_fov_x(w, h, args.fov_x))
    written = export_geometry(
        args.output_dir, rgb, depth, camera, normal, mask,
        args.export, args.stride, args.edge_threshold,
        args.normal_angle_threshold, args.normal_bypass_depth_threshold)
    metadata = {
        "depth_space": "relative" if args.depth_mode == "disparity" else "input",
        "camera": camera.__dict__,
        "exports": {name: str(path) for name, path in written.items()},
    }
    Path(args.output_dir, "geometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
