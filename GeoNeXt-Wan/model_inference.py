import argparse
import os
import sys
from pathlib import Path

# This file is also launched directly by the Wan adapter. Make the release root
# importable so backend-independent helpers under ``utils`` are available.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

from diffsynth.core import load_state_dict
from diffsynth.core.data.data_profiles import DepthNormalProfile
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffusers import AutoencoderKL
from utils.visualization import depth_to_vis, normal_to_vis


def parse_args():
    parser = argparse.ArgumentParser(description="GeoNeXt-Wan multi-target batch inference")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="assets/input",
        help="Directory containing input RGB images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/wan",
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to your trained full checkpoint (*.safetensors).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Sampling steps.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale.")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=0,
        help="Video frame length passed to WAN sampler. Set <=0 to auto-compute from latent frame count.",
    )
    parser.add_argument(
        "--expected_latent_frames",
        type=int,
        default=0,
        help="Minimum latent-time outputs required. Set <=0 to auto: num_condition_frames + len(target_modalities).",
    )
    parser.add_argument("--tiled", action="store_true", help="Enable tiled VAE decode.")
    parser.add_argument("--tile_size_h", type=int, default=30, help="Tile size (height chunks).")
    parser.add_argument("--tile_size_w", type=int, default=52, help="Tile size (width chunks).")
    parser.add_argument("--tile_stride_h", type=int, default=15, help="Tile stride (height chunks).")
    parser.add_argument("--tile_stride_w", type=int, default=26, help="Tile stride (width chunks).")
    parser.add_argument(
        "--processing_res",
        type=int,
        default=768,
        help="Resize the selected image side to this value before inference. Set <=0 to keep input size.",
    )
    parser.add_argument(
        "--processing_res_auto",
        action="store_true",
        help="Choose processing resolution per image from its current size, clamped by min/max.",
    )
    parser.add_argument("--processing_res_min", type=int, default=576, help="Minimum auto processing resolution.")
    parser.add_argument("--processing_res_max", type=int, default=2048, help="Maximum auto processing resolution.")
    parser.add_argument("--processing_res_multiple", type=int, default=64, help="Round auto processing resolution to this multiple.")
    parser.add_argument(
        "--processing_res_side",
        type=str,
        default="long",
        choices=["short", "long"],
        help="Which image side --processing_res targets before inference. long limits the max edge without upscaling.",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="trunc_disparity",
        help="Depth normalization mode used during training (used for depth visualization direction).",
    )
    parser.add_argument(
        "--target_modalities",
        type=str,
        default="depth,normal",
        help="Predicted target order. The release model supports depth,normal.",
    )
    parser.add_argument(
        "--num_condition_frames",
        type=int,
        default=1,
        help="Number of leading condition frames in latent layout.",
    )
    parser.add_argument(
        "--rgb_condition_mode",
        type=str,
        default="first_frame",
        choices=["first_frame", "repeat_add", "concat"],
        help="RGB conditioning mode used by the depth_normal fused path.",
    )
    parser.add_argument("--rgb_condition_scale", type=float, default=1.0, help="Scale used by --rgb_condition_mode repeat_add.")
    parser.add_argument(
        "--train_loss_mode",
        type=str,
        default="flowmatch",
        choices=["flowmatch", "single_step_clean", "single_step_direct_clean"],
        help="Checkpoint training objective mode (for logging/reproducibility hints).",
    )
    parser.add_argument(
        "--single_step_clean_timestep_index",
        type=int,
        default=0,
        help="Training fixed timestep index used by single-step objectives (metadata hint).",
    )
    parser.add_argument(
        "--single_step_zero_noise",
        action="store_true",
        help="Use all-zero sampler noise at inference start. Useful for matching single-step-zero-noise training more closely.",
    )
    parser.add_argument(
        "--temporal_rope_scale",
        type=int,
        default=1,
        help="Temporal RoPE index scale. 1 keeps original WAN behavior; >1 increases temporal separation (e.g., 8 for 3 latent frames).",
    )
    parser.add_argument(
        "--wan_model_dir",
        type=str,
        default="",
        help="Optional WAN base model directory. If empty, use WAN_MODEL_DIR env or built-in default.",
    )
    parser.add_argument("--vae_backend", type=str, default="wan", choices=["wan", "flux"])
    parser.add_argument("--flux_vae_model_name_or_path", type=str, default="")
    parser.add_argument("--flux_vae_subfolder", type=str, default="vae")
    return parser.parse_args()


class FluxVaeBridge(nn.Module):
    """Bridge a diffusers AutoencoderKL to WAN pipeline VAE interface.

    This keeps latent channels unchanged and expects FLUX VAE latent_channels=16.
    """

    def __init__(self, flux_vae: AutoencoderKL, device: torch.device, torch_dtype: torch.dtype):
        super().__init__()
        self.flux_vae = flux_vae
        self.device = device
        self.torch_dtype = torch_dtype
        self.vae_dtype = next(flux_vae.parameters()).dtype
        self.model = type("_FluxVaeMeta", (), {"z_dim": int(getattr(flux_vae.config, "latent_channels", 16))})()
        self.upsampling_factor = 2 ** (len(getattr(flux_vae.config, "block_out_channels", [1, 2, 4, 8])) - 1)
        if self.model.z_dim != 16:
            raise ValueError(
                f"FLUX VAE latent_channels must be 16 for WAN DiT compatibility, got {self.model.z_dim}."
            )

    def encode(self, videos, device=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if isinstance(videos, torch.Tensor):
            seq = [videos[i] for i in range(videos.shape[0])] if videos.dim() == 5 else [videos]
        else:
            seq = videos

        outs = []
        for vid in seq:
            if vid.dim() != 4:
                raise ValueError(f"Expected [C,T,H,W], got {vid.shape}")
            frames = vid.permute(1, 0, 2, 3).to(self.device, dtype=self.vae_dtype)  # [T,C,H,W]
            lat = self.flux_vae.encode(frames).latent_dist.mode()
            lat = lat * self.flux_vae.config.scaling_factor
            outs.append(lat.permute(1, 0, 2, 3).contiguous().to(dtype=self.torch_dtype))  # [16,T,h,w]
        return torch.stack(outs, dim=0)  # [B,16,T,h,w]

    def decode(self, hidden_states, device=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if hidden_states.dim() != 5:
            raise ValueError(f"Expected [B,16,T,h,w], got {hidden_states.shape}")
        outs = []
        for z in hidden_states:
            lat = z.permute(1, 0, 2, 3).contiguous().to(dtype=self.vae_dtype)  # [T,16,h,w]
            lat = lat / self.flux_vae.config.scaling_factor
            rec = self.flux_vae.decode(lat).sample  # [T,3,H,W]
            outs.append(rec.permute(1, 0, 2, 3).contiguous().to(dtype=self.torch_dtype))  # [3,T,H,W]
        return torch.stack(outs, dim=0)  # [B,3,T,H,W]


def configure_depth_normal_inference(pipe, args):
    # Reuse training profile logic to swap in the custom three-frame fused unit/model_fn.
    profile_args = argparse.Namespace(
        norm_type=args.norm_type,
        rgb_condition_mode=args.rgb_condition_mode,
        rgb_condition_scale=args.rgb_condition_scale,
        target_modalities=args.target_modalities,
        num_condition_frames=args.num_condition_frames,
    )
    profile = DepthNormalProfile(profile_args)
    profile.configure_pipeline(pipe)
    pipe.norm_type = args.norm_type
    pipe.dit.temporal_rope_scale = max(1, int(args.temporal_rope_scale))
    print(f"[Config] temporal_rope_scale={pipe.dit.temporal_rope_scale}")


def _resolve_wan_dit_path(wan_model_dir: Path):
    single = wan_model_dir / "diffusion_pytorch_model.safetensors"
    if single.exists():
        return str(single)
    shards = sorted(wan_model_dir.glob("diffusion_pytorch_model-*.safetensors"))
    if len(shards) > 0:
        # ModelPool now auto-expands shard groups from a shard path.
        return str(shards[0])
    raise FileNotFoundError(
        f"No WAN DiT weights found under {wan_model_dir}. "
        "Expected diffusion_pytorch_model.safetensors or diffusion_pytorch_model-*.safetensors"
    )


def _resolve_wan_vae_path(wan_model_dir: Path):
    candidates = [
        "Wan2.2_VAE.pth",
        "Wan2.1_VAE.pth",
        "Wan2.2_VAE.safetensors",
        "Wan2.1_VAE.safetensors",
    ]
    for name in candidates:
        p = wan_model_dir / name
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        f"No WAN VAE file found under {wan_model_dir}. Tried: {candidates}"
    )


def build_pipe(wan_model_dir: str = "", override_vae_path: str = ""):
    resolved_model_dir = str(wan_model_dir).strip() or os.environ.get("WAN_MODEL_DIR", "")
    if not resolved_model_dir:
        raise ValueError(
            "Wan base model is required. Use the unified inference.py entry point "
            "for automatic download, or pass --wan_model_dir."
        )
    wan_model_dir = Path(resolved_model_dir)
    dit_path = _resolve_wan_dit_path(wan_model_dir)
    t5_path = str(wan_model_dir / "models_t5_umt5-xxl-enc-bf16.pth")
    if not Path(t5_path).exists():
        raise FileNotFoundError(f"Missing WAN text encoder: {t5_path}")
    override_vae = str(override_vae_path).strip()
    use_override_vae_repo_id = False
    if override_vae:
        if Path(override_vae).exists():
            vae_path = override_vae
        else:
            # Treat non-existing override as a HuggingFace repo id, e.g. stabilityai/sd-vae-ft-mse.
            # Note: WAN pipeline expects WAN VAE architecture; incompatible repos may still fail at load time.
            vae_path = override_vae
            use_override_vae_repo_id = True
    else:
        vae_path = _resolve_wan_vae_path(wan_model_dir)
    tokenizer_path = str(wan_model_dir / "google" / "umt5-xxl")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(f"Missing WAN tokenizer directory: {tokenizer_path}")

    print(f"[BaseModel] WAN_MODEL_DIR={wan_model_dir}")
    print(f"[BaseModel] DiT={dit_path}")
    print(f"[BaseModel] T5={t5_path}")
    print(f"[BaseModel] VAE={vae_path}")
    if use_override_vae_repo_id:
        print("[BaseModel] VAE override is treated as HuggingFace repo id (origin_file_pattern=diffusion_pytorch_model.safetensors)")

    vae_model_config = (
        ModelConfig(model_id=vae_path, origin_file_pattern="diffusion_pytorch_model.safetensors")
        if use_override_vae_repo_id
        else ModelConfig(path=vae_path)
    )
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(path=dit_path),
            ModelConfig(path=t5_path),
            vae_model_config,
        ],
        tokenizer_config=ModelConfig(path=tokenizer_path),
    )
    return pipe


def build_pipe_for_depth_normal(args):
    pipe = build_pipe(
        args.wan_model_dir,
        getattr(args, "override_vae_path", ""),
    )
    vae_backend = getattr(args, "vae_backend", "wan")
    if vae_backend == "flux":
        flux_vae_path = str(getattr(args, "flux_vae_model_name_or_path", "")).strip()
        if not flux_vae_path:
            raise ValueError("--flux_vae_model_name_or_path is required when --vae_backend=flux")
        flux_vae_subfolder = str(getattr(args, "flux_vae_subfolder", "vae")).strip() or None
        model_dtype = pipe.torch_dtype if torch.cuda.is_available() else torch.float32
        flux_vae = AutoencoderKL.from_pretrained(
            flux_vae_path,
            subfolder=flux_vae_subfolder,
            torch_dtype=model_dtype,
        ).to(pipe.device)
        pipe.vae = FluxVaeBridge(flux_vae=flux_vae, device=pipe.device, torch_dtype=pipe.torch_dtype)
        print(
            f"[BaseModel] Replaced WAN VAE with FLUX VAE backend: model={flux_vae_path}, "
            f"subfolder={flux_vae_subfolder}, z_dim={pipe.vae.model.z_dim}, upsampling_factor={pipe.vae.upsampling_factor}"
        )
    configure_depth_normal_inference(pipe, args)
    return pipe


def _round_up_to_multiple(x, base=16):
    return ((x + base - 1) // base) * base


def _infer_pad_multiple(pipe):
    vae_factor = int(getattr(getattr(pipe, "vae", None), "upsampling_factor", 8))
    patch_size = getattr(getattr(pipe, "dit", None), "patch_size", (1, 2, 2))
    try:
        spatial_patch = int(patch_size[1])
    except Exception:
        spatial_patch = 2
    return max(8, vae_factor * spatial_patch)


def _round_to_multiple(x, base=64):
    base = max(int(base), 1)
    return max(base, int(round(float(x) / base)) * base)


def _resolve_processing_res_for_image(
    w: int,
    h: int,
    processing_res: int,
    processing_res_side: str,
    processing_res_auto: bool = False,
    processing_res_min: int = 576,
    processing_res_max: int = 2048,
    processing_res_multiple: int = 64,
):
    if not processing_res_auto:
        return processing_res

    side = max(w, h) if processing_res_side == "long" else min(w, h)
    if side <= 0:
        return processing_res

    auto_res = _round_to_multiple(side, processing_res_multiple)
    auto_res = max(int(processing_res_min), min(int(processing_res_max), auto_res))
    return auto_res


def _prepare_image_for_inference(
    image: Image.Image,
    processing_res: int,
    pipe,
    processing_res_side: str = "short",
    processing_res_auto: bool = False,
    processing_res_min: int = 576,
    processing_res_max: int = 2048,
    processing_res_multiple: int = 64,
):
    image = image.convert("RGB")
    w, h = image.size
    content_w, content_h = w, h
    processing_res = _resolve_processing_res_for_image(
        w,
        h,
        processing_res,
        processing_res_side,
        processing_res_auto=processing_res_auto,
        processing_res_min=processing_res_min,
        processing_res_max=processing_res_max,
        processing_res_multiple=processing_res_multiple,
    )
    if processing_res is not None and processing_res > 0:
        resize_side = max(w, h) if processing_res_side == "long" else min(w, h)
        should_resize = resize_side > 0 and resize_side != processing_res
        if processing_res_side == "long":
            should_resize = resize_side > processing_res
        if should_resize:
            scale = processing_res / resize_side
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            image = image.resize((new_w, new_h), Image.BILINEAR)
            w, h = image.size
            content_w, content_h = w, h

    pad_multiple = _infer_pad_multiple(pipe)
    safe_w = _round_up_to_multiple(w, pad_multiple)
    safe_h = _round_up_to_multiple(h, pad_multiple)
    if safe_w != w or safe_h != h:
        padded = Image.new("RGB", (safe_w, safe_h), (0, 0, 0))
        padded.paste(image, (0, 0))
        image = padded
    return image, (content_w, content_h)


def _tensor_image_to_pil(image_tensor):
    image = image_tensor.detach().float().cpu()
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    if image.dim() != 3 or image.shape[-1] not in (1, 3):
        return None
    image = image.numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return normal_to_vis(image)


def _depth_tensor_to_pil(depth_tensor, norm_type="trunc_disparity"):
    depth = depth_tensor.detach().float().cpu()
    if depth.dim() == 4:
        depth = depth[0]
    if depth.dim() == 3 and depth.shape[0] >= 1:
        depth = depth[0]
    if depth.dim() != 2:
        return None
    depth_01 = ((depth.clamp(-1, 1) + 1.0) * 0.5).numpy()
    reverse_color = "disparity" in str(norm_type).lower()
    return depth_to_vis(depth_01, reverse_color=reverse_color)


def _modality_tensor_to_pil(frame_tensor, frame_idx, norm_type="trunc_disparity"):
    if frame_idx == 1:
        return _depth_tensor_to_pil(frame_tensor, norm_type=norm_type)
    return _tensor_image_to_pil(frame_tensor)


def _latent_frames_to_video_frames(latent_frames: int):
    latent_frames = max(int(latent_frames), 1)
    return (latent_frames - 1) * 4 + 1


def _parse_target_modalities(text):
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    if len(items) == 0:
        return ["depth", "normal"]
    unsupported = [item for item in items if item not in {"depth", "normal"}]
    if unsupported:
        raise ValueError(f"Unsupported release modalities: {unsupported}. Use depth,normal.")
    # Keep depth first if present to match training-side convention.
    if "depth" in items:
        items = ["depth"] + [x for x in items if x != "depth"]
    return items


def _build_frame_names(num_condition_frames, target_modalities):
    n_cond = max(int(num_condition_frames), 1)
    if n_cond == 1:
        cond_names = ["rgb"]
    else:
        cond_names = [f"rgb_cond{i}" for i in range(n_cond)]
    return cond_names + list(target_modalities)


def _frame_tensor_to_vis(frame_tensor, frame_name, norm_type="trunc_disparity"):
    if frame_name == "depth":
        return _depth_tensor_to_pil(frame_tensor, norm_type=norm_type)
    return _tensor_image_to_pil(frame_tensor)


def main():
    args = parse_args()
    target_modalities = _parse_target_modalities(args.target_modalities)
    frame_names = _build_frame_names(args.num_condition_frames, target_modalities)
    expected_latent_frames = (
        int(args.expected_latent_frames)
        if int(args.expected_latent_frames) > 0
        else len(frame_names)
    )
    num_frames = (
        int(args.num_frames)
        if int(args.num_frames) > 0
        else _latent_frames_to_video_frames(expected_latent_frames)
    )

    print(f"[Config] frame_names={frame_names}")
    print(f"[Config] expected_latent_frames={expected_latent_frames}, num_frames={num_frames}")
    print(
        "[Config] train_loss_mode="
        f"{args.train_loss_mode}, "
        f"single_step_clean_timestep_index={args.single_step_clean_timestep_index}, "
        f"single_step_zero_noise={args.single_step_zero_noise}"
    )
    if args.train_loss_mode == "single_step_direct_clean" and args.num_inference_steps != 1:
        print(
            "[Warn] Checkpoint was trained with single_step_direct_clean. "
            f"You are running num_inference_steps={args.num_inference_steps}. "
            "For closest train/infer behavior, try --num_inference_steps 1."
        )
    if args.single_step_zero_noise and args.num_inference_steps != 1:
        print(
            "[Warn] --single_step_zero_noise is usually only meaningful with "
            "--num_inference_steps 1."
        )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    vis_dirs = {}
    for name in frame_names:
        vis_dir_name = "rgb_recon" if name == "rgb" else f"{name}_vis"
        vis_dirs[name] = output_dir / vis_dir_name
    for folder in vis_dirs.values():
        folder.mkdir(parents=True, exist_ok=True)
    depth_raw_dir = output_dir / "depth_raw"
    if "depth" in frame_names:
        depth_raw_dir.mkdir(parents=True, exist_ok=True)
    normal_raw_dir = output_dir / "normal_raw"
    if "normal" in frame_names:
        normal_raw_dir.mkdir(parents=True, exist_ok=True)

    image_exts = {".png", ".jpg", ".jpeg"}
    image_paths = sorted(p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts)
    if len(image_paths) == 0:
        raise ValueError(f"No images found under: {input_dir}")

    print(f"Found {len(image_paths)} images in {input_dir}")
    pipe = build_pipe_for_depth_normal(args)

    state_dict = load_state_dict(args.checkpoint)
    load_result = pipe.dit.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint: {args.checkpoint}")
    if len(load_result.unexpected_keys) > 0:
        print(f"[Warn] unexpected_keys: {len(load_result.unexpected_keys)}")
    if len(load_result.missing_keys) > 0:
        print(f"[Warn] missing_keys: {len(load_result.missing_keys)}")

    tile_size = (args.tile_size_h, args.tile_size_w)
    tile_stride = (args.tile_stride_h, args.tile_stride_w)

    for image_path in tqdm(image_paths, desc="Inference"):
        # Align image loading behavior with MoGe: cv2.imread (BGR) -> RGB.
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[Warn] failed to read image, skip: {image_path}")
            continue
        rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ori = Image.fromarray(rgb_np, mode="RGB")
        ori_w, ori_h = ori.size
        rgb, (content_w, content_h) = _prepare_image_for_inference(
            ori,
            args.processing_res,
            pipe,
            args.processing_res_side,
            processing_res_auto=args.processing_res_auto,
            processing_res_min=args.processing_res_min,
            processing_res_max=args.processing_res_max,
            processing_res_multiple=args.processing_res_multiple,
        )
        h, w = rgb.size[1], rgb.size[0]
        print(
            f"[Size] {image_path.name}: "
            f"ori=({ori_h},{ori_w}) "
            f"content_after_resize=({content_h},{content_w}) "
            f"padded_input=({h},{w})"
        )

        # Input-image-only inference: fused unit uses the first frame as condition.
        input_video = [rgb]
        latents = pipe(
            prompt="",
            negative_prompt="",
            input_video=input_video,
            seed=args.seed,
            rand_device="cuda",
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            num_frames=num_frames,
            height=h,
            width=w,
            tiled=args.tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
            zero_noise=args.single_step_zero_noise,
            direct_clean_output=(args.train_loss_mode == "single_step_direct_clean"),
            output_type="latent",
        )

        if latents.shape[2] < expected_latent_frames:
            raise RuntimeError(
                f"Expected at least {expected_latent_frames} latent frames, got {latents.shape[2]} for {image_path.name}. "
                "Check --num_frames / --expected_latent_frames."
            )
        print(f"[Size] {image_path.name}: latents_shape={tuple(latents.shape)}")
        # import ipdb; ipdb.set_trace()  # --- DEBUG ---
        pipe.load_models_to_device(["vae"])
        decoded_frames = []
        raw_depth = None
        raw_normal = None
        for idx in range(expected_latent_frames):
            single_latent = latents[:, :, idx:idx + 1, :, :]
            decoded = pipe.vae.decode(
                single_latent,
                device=pipe.device,
                tiled=args.tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            if decoded.shape[2] != 1:
                raise RuntimeError(
                    f"Expected one decoded frame per latent slice, got T={decoded.shape[2]} at latent index {idx} for {image_path.name}"
                )
            frame_name = frame_names[idx] if idx < len(frame_names) else f"frame_{idx}"
            if frame_name == "depth":
                # Save the prediction before colorization. With trunc_disparity
                # this is normalized inverse depth rather than metric depth.
                raw_depth = decoded[0, 0, 0].detach().float().cpu().numpy()
            elif frame_name == "normal":
                raw_normal = decoded[0, :, 0].detach().float().cpu().permute(1, 2, 0).numpy()
            vis_img = _frame_tensor_to_vis(decoded[0, :, 0], frame_name, norm_type=args.norm_type)
            decoded_frames.append(vis_img)
        pipe.load_models_to_device([])

        stem = image_path.stem
        if raw_depth is not None:
            raw_depth = raw_depth[:content_h, :content_w]
            raw_depth = cv2.resize(raw_depth, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            # Public output contract: normalized disparity is always [0, 1].
            # The VAE decoder emits image-space values in [-1, 1].
            raw_depth = np.clip((raw_depth + 1.0) * 0.5, 0.0, 1.0)
            np.save(depth_raw_dir / f"{stem}.npy", raw_depth.astype(np.float32))
        if raw_normal is not None:
            raw_normal = raw_normal[:content_h, :content_w]
            raw_normal = cv2.resize(raw_normal, (ori_w, ori_h), interpolation=cv2.INTER_LINEAR)
            raw_normal = np.clip(raw_normal, -1.0, 1.0)
            np.save(normal_raw_dir / f"{stem}.npy", raw_normal.astype(np.float32))
        # Save per-frame visualization.
        for idx in range(expected_latent_frames):
            frame_name = frame_names[idx] if idx < len(frame_names) else f"frame_{idx}"
            vis_img = decoded_frames[idx]

            if vis_img is not None:
                vis_img = vis_img.crop((0, 0, content_w, content_h)).resize((ori_w, ori_h), Image.BILINEAR)
                vis_img.save(vis_dirs[frame_name] / f"{stem}.png")

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
