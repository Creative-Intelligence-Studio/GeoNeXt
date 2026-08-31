from .base_pipeline import BasePipeline
import torch
import os
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from diffsynth.core.data.custom_operators import colorize_depth_map


def _save_debug_grid(pipe, labels, tiles, output_name):
    tiles = [(label, tile) for label, tile in zip(labels, tiles) if isinstance(tile, Image.Image)]
    if len(tiles) == 0:
        return None

    tile_width = max(tile.size[0] for _, tile in tiles)
    tile_height = max(tile.size[1] for _, tile in tiles)
    canvas = Image.new("RGB", (tile_width * len(tiles), tile_height + 32), "white")
    drawer = ImageDraw.Draw(canvas)

    for idx, (label, tile) in enumerate(tiles):
        x_offset = idx * tile_width
        normalized_tile = ImageOps.pad(tile.convert("RGB"), (tile_width, tile_height), color="black")
        canvas.paste(normalized_tile, (x_offset, 32))
        drawer.text((x_offset + 8, 8), label, fill="black")

    debug_dir = getattr(pipe, "debug_visualize_dir", None) or os.path.join(os.getcwd(), "debug_inputs")
    os.makedirs(debug_dir, exist_ok=True)
    output_path = os.path.abspath(os.path.join(debug_dir, output_name))
    canvas.save(output_path)
    return output_path


def _tensor_image_to_pil(image_tensor):
    if not isinstance(image_tensor, torch.Tensor):
        return None
    image = image_tensor.detach().float().cpu()
    if image.dim() == 4:
        image = image[0]
    if image.dim() == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    if image.dim() != 3 or image.shape[-1] not in (1, 3):
        return None
    image = ((image.clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8).numpy()
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    return Image.fromarray(image, mode="RGB")


def _depth_tensor_to_pil(depth_tensor, norm_type="trunc_disparity"):
    if not isinstance(depth_tensor, torch.Tensor):
        return None
    depth = depth_tensor.detach().float().cpu()
    if depth.dim() == 4:
        depth = depth[0]
    if depth.dim() == 3 and depth.shape[0] >= 1:
        depth = depth[0]
    if depth.dim() != 2:
        return None
    depth_01 = ((depth.clamp(-1, 1) + 1.0) * 0.5).numpy()
    reverse_color = "disparity" in str(norm_type).lower()
    return colorize_depth_map(depth_01, reverse_color=reverse_color)


def _modality_tensor_to_pil(frame_tensor, frame_idx, norm_type="trunc_disparity"):
    if frame_idx == 1:
        return _depth_tensor_to_pil(frame_tensor, norm_type=norm_type)
    return _tensor_image_to_pil(frame_tensor)


def _decode_latents_to_frames(pipe, latents, tiled=False, tile_size=None, tile_stride=None):
    decoded_frames = []
    pipe.load_models_to_device(["vae"])
    with torch.no_grad():
        for frame_idx in range(latents.shape[2]):
            single_latent = latents[:, :, frame_idx:frame_idx + 1, :, :]
            decoded = pipe.vae.decode(
                single_latent,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
            decoded_frames.append(decoded[0, :, 0].detach())
    return decoded_frames


def _maybe_debug_visualize_loss_outputs(pipe, inputs, timestep, noise_pred, training_target):
    every = getattr(pipe, "debug_visualize_every", 0)
    step = getattr(pipe, "_debug_forward_step", 0)
    if every <= 0:
        return
    if step != 1 and step % every != 0:
        return

    latents = inputs["latents"].detach()
    input_latents = inputs["input_latents"].detach()
    sigma = pipe.scheduler.sigmas[torch.argmin((pipe.scheduler.timesteps - timestep.cpu()).abs())].to(device=latents.device, dtype=latents.dtype)
    pred_clean_latents = latents - sigma * noise_pred.detach().to(dtype=latents.dtype)
    target_clean_latents = input_latents
    target_noise_latents = latents - sigma * training_target.detach().to(dtype=latents.dtype)

    pred_frames = _decode_latents_to_frames(
        pipe, pred_clean_latents,
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )
    target_frames = _decode_latents_to_frames(
        pipe, target_clean_latents,
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )
    noise_target_frames = _decode_latents_to_frames(
        pipe, target_noise_latents,
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )

    labels = []
    tiles = []
    norm_type = getattr(pipe, "norm_type", "trunc_disparity")
    for idx in range(min(3, len(pred_frames))):
        labels.extend([f"pred_clean[{idx}]", f"target_clean[{idx}]", f"target_from_gt_noise[{idx}]"])
        tiles.extend([
            _modality_tensor_to_pil(pred_frames[idx], idx, norm_type=norm_type),
            _modality_tensor_to_pil(target_frames[idx], idx, norm_type=norm_type),
            _modality_tensor_to_pil(noise_target_frames[idx], idx, norm_type=norm_type),
        ])

    debug_path = _save_debug_grid(pipe, labels, tiles, f"wan_model_pred_layout_step_{step:06d}.png")
    print(f"[Debug] Saved model-prediction visualization to {debug_path}")


def _maybe_debug_visualize_direct_clean_outputs(pipe, inputs, pred_clean_latents):
    every = getattr(pipe, "debug_visualize_every", 0)
    step = getattr(pipe, "_debug_forward_step", 0)
    if every <= 0:
        return
    if step != 1 and step % every != 0:
        return

    pred_frames = _decode_latents_to_frames(
        pipe, pred_clean_latents.detach(),
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )
    target_frames = _decode_latents_to_frames(
        pipe, inputs["input_latents"].detach(),
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )

    labels = []
    tiles = []
    norm_type = getattr(pipe, "norm_type", "trunc_disparity")
    for idx in range(min(3, len(pred_frames), len(target_frames))):
        labels.extend([f"pred_direct_clean[{idx}]", f"target_clean[{idx}]"])
        tiles.extend([
            _modality_tensor_to_pil(pred_frames[idx], idx, norm_type=norm_type),
            _modality_tensor_to_pil(target_frames[idx], idx, norm_type=norm_type),
        ])

    debug_path = _save_debug_grid(pipe, labels, tiles, f"wan_model_pred_direct_clean_layout_step_{step:06d}.png")
    print(f"[Debug] Saved direct-clean prediction visualization to {debug_path}")


def _maybe_debug_visualize_single_step_clean_outputs(pipe, inputs, pred_clean_latents):
    every = getattr(pipe, "debug_visualize_every", 0)
    step = getattr(pipe, "_debug_forward_step", 0)
    if every <= 0:
        return
    if step != 1 and step % every != 0:
        return

    pred_frames = _decode_latents_to_frames(
        pipe, pred_clean_latents.detach(),
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )
    target_frames = _decode_latents_to_frames(
        pipe, inputs["input_latents"].detach(),
        tiled=inputs.get("tiled", False),
        tile_size=inputs.get("tile_size"),
        tile_stride=inputs.get("tile_stride"),
    )

    labels = []
    tiles = []
    norm_type = getattr(pipe, "norm_type", "trunc_disparity")
    for idx in range(min(3, len(pred_frames), len(target_frames))):
        labels.extend([f"pred_single_step_clean[{idx}]", f"target_clean[{idx}]"])
        tiles.extend([
            _modality_tensor_to_pil(pred_frames[idx], idx, norm_type=norm_type),
            _modality_tensor_to_pil(target_frames[idx], idx, norm_type=norm_type),
        ])

    debug_path = _save_debug_grid(pipe, labels, tiles, f"wan_model_pred_single_step_clean_layout_step_{step:06d}.png")
    print(f"[Debug] Saved single-step-clean prediction visualization to {debug_path}")


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    _maybe_debug_visualize_loss_outputs(pipe, inputs, timestep, noise_pred, training_target)

    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none')
    
    if "loss_mask" in inputs:
        mask = inputs["loss_mask"]
        loss = loss * mask
        
        loss = loss.sum() / (mask.expand_as(loss).sum() + 1e-6)
    else:
        loss = loss.mean()
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def SingleStepCleanSFTLoss(pipe: BasePipeline, **inputs):
    # Fixed single-step clean-data objective:
    # 1) build one noisy latent state at a fixed training timestep index
    # 2) predict model output once
    # 3) convert prediction to clean-latent estimate and supervise in clean space
    timestep_index = int(inputs.get("single_step_clean_timestep_index", 0))
    timestep_index = max(0, min(timestep_index, len(pipe.scheduler.timesteps) - 1))
    timestep = pipe.scheduler.timesteps[timestep_index].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

    use_zero_noise = bool(inputs.get("single_step_zero_noise", False))
    noise = torch.zeros_like(inputs["input_latents"]) if use_zero_noise else torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    if bool(inputs.get("single_step_zero_target_latents", False)):
        n_cond = 0
        if "clean_reference_latents" in inputs and isinstance(inputs["clean_reference_latents"], torch.Tensor):
            n_cond = int(inputs["clean_reference_latents"].shape[2])
        n_cond = max(0, min(n_cond, int(inputs["latents"].shape[2])))
        if n_cond < int(inputs["latents"].shape[2]):
            inputs["latents"][:, :, n_cond:, :, :] = 0

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)

    sigma = pipe.scheduler.sigmas[timestep_index].to(device=inputs["latents"].device, dtype=inputs["latents"].dtype)
    pred_clean_latents = inputs["latents"] - sigma * noise_pred
    _maybe_debug_visualize_single_step_clean_outputs(pipe, inputs, pred_clean_latents)

    loss = torch.nn.functional.mse_loss(pred_clean_latents.float(), inputs["input_latents"].float(), reduction="none")
    if "loss_mask" in inputs:
        mask = inputs["loss_mask"]
        loss = loss * mask
        loss = loss.sum() / (mask.expand_as(loss).sum() + 1e-6)
    else:
        loss = loss.mean()
    return loss


def SingleStepDirectCleanLatentSFTLoss(pipe: BasePipeline, **inputs):
    # Fixed single-step objective with direct clean-latent supervision:
    # model output is supervised directly by clean latents.
    timestep_index = int(inputs.get("single_step_clean_timestep_index", 0))
    timestep_index = max(0, min(timestep_index, len(pipe.scheduler.timesteps) - 1))
    timestep = pipe.scheduler.timesteps[timestep_index].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

    use_zero_noise = bool(inputs.get("single_step_zero_noise", False))
    noise = torch.zeros_like(inputs["input_latents"]) if use_zero_noise else torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    if bool(inputs.get("single_step_zero_target_latents", False)):
        n_cond = 0
        if "clean_reference_latents" in inputs and isinstance(inputs["clean_reference_latents"], torch.Tensor):
            n_cond = int(inputs["clean_reference_latents"].shape[2])
        n_cond = max(0, min(n_cond, int(inputs["latents"].shape[2])))
        if n_cond < int(inputs["latents"].shape[2]):
            inputs["latents"][:, :, n_cond:, :, :] = 0

    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    model_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    _maybe_debug_visualize_direct_clean_outputs(pipe, inputs, model_pred)

    loss = torch.nn.functional.mse_loss(model_pred.float(), inputs["input_latents"].float(), reduction="none")
    if "loss_mask" in inputs:
        mask = inputs["loss_mask"]
        loss = loss * mask
        loss = loss.sum() / (mask.expand_as(loss).sum() + 1e-6)
    else:
        loss = loss.mean()
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss


class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
