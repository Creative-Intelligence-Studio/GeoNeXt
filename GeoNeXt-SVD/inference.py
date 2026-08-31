"""GeoNeXt-SVD inference adapter."""

from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image

from utils.common import list_images, safe_stem
from utils.visualization import save_prediction_outputs


GEONEXT_REPO = "happy0612/GeoNeXt"
SVD_BASE_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"


def _resolve_checkpoint(checkpoint):
    if checkpoint is not None:
        return checkpoint
    from huggingface_hub import snapshot_download

    print("Downloading GeoNeXt-SVD checkpoint from %s" % GEONEXT_REPO)
    root = snapshot_download(repo_id=GEONEXT_REPO, allow_patterns="GeoNeXt-SVD/**")
    return str(Path(root) / "GeoNeXt-SVD")


def _resize(image, target, side):
    if not target or target <= 0:
        return image
    width, height = image.size
    current = max(width, height) if side == "long" else min(width, height)
    scale = target / float(current)
    return image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.BILINEAR)


def run(args):
    import torch
    from diffusers import AutoencoderKL, UNetSpatioTemporalConditionModel
    from pipeline import GeoNeXtPipeline
    from utils.geometry import Camera, disparity_to_relative_depth, export_geometry

    processing_res = 768 if args.processing_res is None else args.processing_res
    processing_side = args.processing_res_side or "long"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.half_precision and device.type == "cuda" else torch.float32
    variant = "fp16" if dtype == torch.float16 else None
    checkpoint = _resolve_checkpoint(args.checkpoint)
    base_model = args.base_model or SVD_BASE_MODEL
    vae = AutoencoderKL.from_pretrained(args.vae_model, torch_dtype=dtype, subfolder=None)
    unet = UNetSpatioTemporalConditionModel.from_pretrained(
        checkpoint, subfolder="unet", torch_dtype=dtype, low_cpu_mem_usage=False)
    pipe = GeoNeXtPipeline.from_pretrained(
        base_model, unet=unet, vae=vae, variant=variant,
        torch_dtype=dtype, low_cpu_mem_usage=False).to(device)
    pipe.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    root, images = list_images(args.input)
    output_root = Path(args.output)
    aligner = None
    if args.align_space == "moge":
        from utils.moge import MoGeAligner
        aligner = MoGeAligner(args.moge_model)

    for image_path in images:
        original = Image.open(image_path).convert("RGB")
        image = _resize(original, processing_res, processing_side)
        width, height = image.size
        width, height = max(64, round(width / 64) * 64), max(64, round(height / 64) * 64)
        image = image.resize((width, height), Image.Resampling.BICUBIC)
        context = torch.autocast("cuda", dtype=torch.float16) if dtype == torch.float16 else nullcontext()
        with torch.no_grad(), context:
            pred = pipe(image, num_frames=3, width=width, height=height,
                        min_guidance_scale=1.0, max_guidance_scale=1.2,
                        noise_aug_strength=0.0, decode_chunk_size=8, generator=generator,
                        motion_bucket_id=127, fps=7, num_inference_steps=args.steps)
        depth = pred.geo_res[0].mean(dim=1).squeeze().float().cpu().numpy()
        normal = pred.geo_res[1].squeeze().permute(1, 2, 0).float().cpu().numpy()
        target_size = original.size
        depth = np.asarray(Image.fromarray(depth, mode="F").resize(target_size, Image.Resampling.BILINEAR))
        normal = np.stack([np.asarray(Image.fromarray(normal[..., i], mode="F").resize(
            target_size, Image.Resampling.BILINEAR)) for i in range(3)], -1)
        stem = safe_stem(image_path, root)
        save_prediction_outputs(output_root, stem, original, depth, normal)
        if args.export:
            case = output_root / "geometry" / stem
            case.mkdir(parents=True, exist_ok=True)
            mask = None
            if aligner is not None:
                relative_depth, mask, camera, intrinsics, scale, shift = aligner.align_disparity(depth, original)
                if not args.use_mask:
                    mask = None
                np.save(case / "intrinsics.npy", intrinsics)
            else:
                relative_depth = disparity_to_relative_depth(depth)
                camera = Camera.from_fov_x(relative_depth.shape[1], relative_depth.shape[0], args.fov_x)
            np.save(case / "geometry_depth.npy", relative_depth.astype(np.float32))
            export_geometry(case, original, relative_depth, camera, normal=normal, mask=mask,
                            exports=args.export, stride=args.stride, edge_threshold=args.edge_threshold,
                            normal_angle_threshold=args.normal_angle_threshold,
                            normal_bypass_depth_threshold=args.normal_bypass_depth_threshold)
        print("saved: %s" % stem)
