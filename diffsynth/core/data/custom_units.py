import torch
import torch.nn.functional as F
from diffsynth.pipelines.wan_video import WanVideoPipeline
from diffsynth.diffusion.base_pipeline import PipelineUnit
from PIL import Image


def _frame_to_vae_video_tensor(pipe: WanVideoPipeline, frame):
    if isinstance(frame, Image.Image):
        return pipe.preprocess_video([frame])

    if not torch.is_tensor(frame):
        raise TypeError(f"Unsupported frame type for VAE encoding: {type(frame)}")

    tensor = frame
    if tensor.dim() == 3:
        # (C, H, W) -> (1, C, 1, H, W)
        if tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1).contiguous()
        tensor = tensor.unsqueeze(0).unsqueeze(2)
    elif tensor.dim() == 4:
        # Accept (1, C, H, W) or (C, T, H, W)
        if tensor.shape[0] == 1 and tensor.shape[1] in (1, 3):
            tensor = tensor.unsqueeze(2)
        elif tensor.shape[0] in (1, 3):
            tensor = tensor.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected 4D frame tensor shape: {tuple(tensor.shape)}")
    elif tensor.dim() == 5:
        pass
    else:
        raise ValueError(f"Unsupported frame tensor rank: {tensor.dim()}")

    return tensor.to(dtype=pipe.torch_dtype, device=pipe.device)
# custom_units.py
class WanVideoUnit_GeoNeXtFused(PipelineUnit):
    """
    1. Receives noise from NoiseInitializer.
    2. Receives input_video (multi-frame).
    3. Outputs input_latents (clean GT latents) for loss computation.
    4. Outputs latents (noisy latents) for model_fn input.
    5. Outputs clean_reference_latents used by model_fn for frame replacement.
    """
    def __init__(self, num_condition_frames=1, rgb_condition_mode="first_frame", rgb_condition_scale=1.0, loss_mask_mode="depth_only"):
        super().__init__(
            
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "valid_mask", "sky_mask", "target_frame_loss_mask"),
            output_params=("latents", "input_latents", "clean_reference_latents", "rgb_condition_latents", "rgb_condition_mode", "rgb_condition_scale"), 
            onload_model_names=("vae",)
        )
        self.num_condition_frames = num_condition_frames
        self.rgb_condition_mode = rgb_condition_mode
        self.rgb_condition_scale = float(rgb_condition_scale)
        self.loss_mask_mode = loss_mask_mode

    def _downsample_mask(self, mask, latent_h, latent_w):
        if mask is None:
            return None
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(0)
        mask = mask.float()
        invalid_mask = ~mask.bool()
        src_h, src_w = invalid_mask.shape[-2:]
        scale_h = max(src_h // latent_h, 1)
        scale_w = max(src_w // latent_w, 1)
        if src_h % latent_h == 0 and src_w % latent_w == 0:
            invalid_down = F.max_pool2d(invalid_mask.float(), kernel_size=(scale_h, scale_w), stride=(scale_h, scale_w))
        else:
            invalid_down = F.interpolate(invalid_mask.float(), size=(latent_h, latent_w), mode="nearest")
        return (~invalid_down.bool()).to(mask.device)

    def _build_loss_mask(self, input_latents, valid_mask=None, sky_mask=None, target_frame_loss_mask=None):
        bsz, channels, frames, latent_h, latent_w = input_latents.shape
        loss_mask = torch.zeros((bsz, channels, frames, latent_h, latent_w), device=input_latents.device, dtype=input_latents.dtype)

        default_mask = torch.ones((1, 1, latent_h, latent_w), device=input_latents.device, dtype=torch.bool)
        valid_down = self._downsample_mask(valid_mask, latent_h, latent_w) if valid_mask is not None else default_mask
        sky_down = self._downsample_mask(sky_mask, latent_h, latent_w) if sky_mask is not None else None

        for frame_idx in range(frames):
            frame_mask = default_mask
            if frame_idx >= self.num_condition_frames:
                if self.loss_mask_mode == "all_targets":
                    frame_mask = valid_down
                    if sky_down is not None:
                        frame_mask = frame_mask | sky_down
                elif self.loss_mask_mode == "depth_only":
                    # Backward-compatible behavior: only apply geometric masks on
                    # the first predicted frame (typically depth).
                    if frame_idx == self.num_condition_frames:
                        frame_mask = valid_down
                        if sky_down is not None:
                            frame_mask = frame_mask | sky_down
                else:
                    raise ValueError(f"Unsupported loss_mask_mode: {self.loss_mask_mode}")
            loss_mask[:, :, frame_idx, :, :] = frame_mask.to(dtype=input_latents.dtype)

        # Optional source-specific supervision gating for predicted target frames.
        # target_frame_loss_mask is expected to align with frames after condition frames.
        if target_frame_loss_mask is not None:
            tfm = target_frame_loss_mask
            if isinstance(tfm, torch.Tensor):
                tfm = tfm.to(device=input_latents.device, dtype=input_latents.dtype).flatten()
            else:
                tfm = torch.tensor(tfm, device=input_latents.device, dtype=input_latents.dtype).flatten()
            n_pred = max(frames - self.num_condition_frames, 0)
            if tfm.numel() > 0 and n_pred > 0:
                n = min(n_pred, int(tfm.numel()))
                gate = tfm[:n].view(1, 1, n, 1, 1)
                loss_mask[:, :, self.num_condition_frames:self.num_condition_frames + n, :, :] *= gate
        return loss_mask

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, valid_mask=None, sky_mask=None, target_frame_loss_mask=None):
        if input_video is None or len(input_video) == 0:
            return {}
        if pipe.scheduler.training and len(input_video) < 2:
            raise ValueError(f"Training expects at least 2-frame input_video, got {len(input_video)}")
        if not pipe.scheduler.training and len(input_video) < self.num_condition_frames:
            raise ValueError(f"Inference expects at least {self.num_condition_frames} input frames, got {len(input_video)}")
        if pipe.scheduler.training and not (1 <= self.num_condition_frames < len(input_video)):
            raise ValueError(f"num_condition_frames must be in [1, {len(input_video) - 1}], got {self.num_condition_frames}")
            
        pipe.load_models_to_device(self.onload_model_names)

        if not pipe.scheduler.training:
            cond_latents = []
            for img in input_video[:self.num_condition_frames]:
                video_tensor = _frame_to_vae_video_tensor(pipe, img)
                z = pipe.vae.encode(
                    video_tensor,
                    device=pipe.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride
                ).to(dtype=pipe.torch_dtype, device=pipe.device)
                cond_latents.append(z)
            clean_ref = torch.cat(cond_latents, dim=2)
            return {
                "latents": noise,
                "clean_reference_latents": clean_ref,
                "rgb_condition_latents": clean_ref[:, :, :1],
                "rgb_condition_mode": self.rgb_condition_mode,
                "rgb_condition_scale": self.rgb_condition_scale,
            }
        
        encoded_latents_list = []

        for img in input_video:
            video_tensor = _frame_to_vae_video_tensor(pipe, img)

            z = pipe.vae.encode(
                video_tensor, 
                device=pipe.device, 
                tiled=tiled, 
                tile_size=tile_size, 
                tile_stride=tile_stride
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            
            encoded_latents_list.append(z)

        full_clean_latents = torch.cat(encoded_latents_list, dim=2)
        
        clean_ref = torch.cat(encoded_latents_list[:self.num_condition_frames], dim=2)
        rgb_condition = encoded_latents_list[0]

        mask = self._build_loss_mask(
            full_clean_latents,
            valid_mask=valid_mask,
            sky_mask=sky_mask,
            target_frame_loss_mask=target_frame_loss_mask,
        )

        if pipe.scheduler.training:
            return {
                "latents": noise,                 
                "input_latents": full_clean_latents, 
                "clean_reference_latents": clean_ref,
                "rgb_condition_latents": rgb_condition,
                "rgb_condition_mode": self.rgb_condition_mode,
                "rgb_condition_scale": self.rgb_condition_scale,
                "loss_mask": mask 
            }
        else:
            return {
                "latents": noise,
                "clean_reference_latents": clean_ref,
                "rgb_condition_latents": clean_ref[:, :, :1],
                "rgb_condition_mode": self.rgb_condition_mode,
                "rgb_condition_scale": self.rgb_condition_scale,
            }


# Backward-compatible alias.
WanVideoUnit_ThreeFrameFused = WanVideoUnit_GeoNeXtFused
