"""Unified inference entry point for GeoNeXt-Wan and GeoNeXt-SVD."""

import argparse
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def build_parser():
    parser = argparse.ArgumentParser(description="GeoNeXt monocular geometry inference")
    parser.add_argument("--backend", default="wan", choices=["wan", "svd"])
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Local GeoNeXt checkpoint. Downloads happy0612/GeoNeXt automatically when omitted.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Local base model directory or Hugging Face ID. Uses the backend default when omitted.",
    )
    parser.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse", help="SVD VAE only")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--processing-res", type=int, default=None)
    parser.add_argument("--processing-res-side", choices=["long", "short"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--half-precision", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--export", nargs="*", default=[],
                        choices=["mesh"])
    parser.add_argument("--fov-x", type=float, default=60.0)
    parser.add_argument(
        "--align-space",
        choices=["relative", "moge"],
        default=None,
        help="Mesh depth space. Defaults to moge with --export mesh, otherwise relative.",
    )
    parser.add_argument("--moge-model", default="Ruicheng/moge-2-vits-normal")
    parser.add_argument("--use-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--edge-threshold", type=float, default=0.04)
    parser.add_argument("--normal-angle-threshold", type=float, default=70.0)
    parser.add_argument("--normal-bypass-depth-threshold", type=float, default=0.015)
    return parser


def load_backend(name):
    path = ROOT / ("GeoNeXt-Wan" if name == "wan" else "GeoNeXt-SVD") / "inference.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("geonext_%s_backend" % name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    args = build_parser().parse_args()
    if args.align_space is None:
        args.align_space = "moge" if args.export else "relative"
    load_backend(args.backend).run(args)


if __name__ == "__main__":
    main()
