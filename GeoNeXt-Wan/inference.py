"""GeoNeXt-Wan inference adapter."""

import subprocess
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from utils.common import list_images


GEONEXT_REPO = "happy0612/GeoNeXt"
GEONEXT_WAN_FILE = "GeoNeXt-Wan/geonext_wan.safetensors"
WAN_BASE_MODEL = "Wan-AI/Wan2.1-T2V-1.3B"


def _resolve_models(args):
    from huggingface_hub import hf_hub_download, snapshot_download

    checkpoint = args.checkpoint
    if checkpoint is None:
        print("Downloading GeoNeXt-Wan checkpoint from %s" % GEONEXT_REPO)
        checkpoint = hf_hub_download(repo_id=GEONEXT_REPO, filename=GEONEXT_WAN_FILE)

    base_model = args.base_model or WAN_BASE_MODEL
    if not Path(base_model).exists():
        print("Downloading Wan base model from %s" % base_model)
        base_model = snapshot_download(repo_id=base_model)
    return checkpoint, base_model


def run(args):
    from utils.geometry import Camera, disparity_to_relative_depth, export_geometry

    aligner = None
    if args.align_space == "moge":
        from utils.moge import MoGeAligner
        aligner = MoGeAligner(args.moge_model)
    root, images = list_images(args.input)
    checkpoint, base_model = _resolve_models(args)
    processing_res = 768 if args.processing_res is None else args.processing_res
    processing_side = args.processing_res_side or "long"
    temporary_input = None
    input_dir = root
    if Path(args.input).is_file():
        # The underlying Wan batch runner consumes a directory. Stage a single
        # input transparently so the public CLI accepts the same inputs as SVD.
        temporary_input = tempfile.TemporaryDirectory(prefix="geonext-wan-input-")
        input_dir = Path(temporary_input.name)
        shutil.copy2(args.input, input_dir / Path(args.input).name)
    try:
        command = [
            sys.executable, str(Path(__file__).with_name("model_inference.py")),
            "--checkpoint", checkpoint,
            "--wan_model_dir", base_model,
            "--input_dir", str(input_dir),
            "--output_dir", args.output,
            "--target_modalities", "depth,normal",
            "--rgb_condition_mode", "concat",
            "--num_inference_steps", str(args.steps),
            "--cfg_scale", "1.0",
            "--processing_res", str(processing_res),
            "--processing_res_side", processing_side,
            "--temporal_rope_scale", "8",
            "--seed", str(args.seed),
        ]
        subprocess.run(command, check=True)
    finally:
        if temporary_input is not None:
            temporary_input.cleanup()
    if not args.export:
        return

    output = Path(args.output)
    for image_path in images:
        stem = image_path.stem
        depth_path = output / "depth_raw" / (stem + ".npy")
        normal_raw_path = output / "normal_raw" / (stem + ".npy")
        normal_path = output / "normal_vis" / (stem + ".png")
        if not depth_path.exists() or (not normal_raw_path.exists() and not normal_path.exists()):
            print("warning: missing prediction for %s; geometry skipped" % image_path)
            continue
        disparity = np.load(depth_path)
        if normal_raw_path.exists():
            normal = np.load(normal_raw_path).astype(np.float32)
        else:
            normal = np.asarray(Image.open(normal_path).convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        rgb = Image.open(image_path).convert("RGB")
        mask = None
        if aligner is not None:
            depth, mask, camera, intrinsics, scale, shift = aligner.align_disparity(disparity, rgb)
            if not args.use_mask:
                mask = None
        else:
            depth = disparity_to_relative_depth(disparity)
            camera = Camera.from_fov_x(depth.shape[1], depth.shape[0], args.fov_x)
        case = output / "geometry" / stem
        case.mkdir(parents=True, exist_ok=True)
        np.save(case / "depth.npy", depth)
        if aligner is not None:
            np.save(case / "intrinsics.npy", intrinsics)
        export_geometry(case, rgb, depth, camera,
                        normal=normal, mask=mask, exports=args.export, stride=args.stride,
                        edge_threshold=args.edge_threshold,
                        normal_angle_threshold=args.normal_angle_threshold,
                        normal_bypass_depth_threshold=args.normal_bypass_depth_threshold)
        print("geometry: %s" % case)
