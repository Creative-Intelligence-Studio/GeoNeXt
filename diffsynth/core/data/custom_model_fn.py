import torch
from einops import rearrange
# Import required dependencies while keeping compatibility with upstream logic.
from diffsynth.diffusion import FlowMatchScheduler
from diffsynth.core import ModelConfig, gradient_checkpoint_forward
from diffsynth.diffusion.base_pipeline import BasePipeline, PipelineUnit

from typing import Optional, Any, Dict, List
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
from diffsynth.models.wan_video_dit_s2v import rope_precompute
from diffsynth.models.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer
from diffsynth.models.wan_video_vae import WanVideoVAE
from diffsynth.models.wan_video_image_encoder import WanImageEncoder
from diffsynth.models.wan_video_vace import VaceWanModel
from diffsynth.models.wan_video_motion_controller import WanMotionControllerModel
from diffsynth.models.wan_video_animate_adapter import WanAnimateAdapter
from diffsynth.models.wan_video_mot import MotWanModel
from diffsynth.models.wav2vec import WanS2VAudioEncoder
from diffsynth.pipelines.wan_video import TeaCache,TemporalTiler_BCTHW, model_fn_wans2v

try:
    from diffsynth.models.longcat_video_dit import LongCatVideoTransformer3DModel
    from diffsynth.pipelines.wan_video import model_fn_longcat_video
except Exception:
    LongCatVideoTransformer3DModel = None
    model_fn_longcat_video = None

def model_fn_wan_video_geonext(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    vap: MotWanModel = None,
    animate_adapter: WanAnimateAdapter = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    vap_hidden_state = None,
    vap_clip_feature = None,
    context_vap = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    longcat_latents=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    # Added argument: clean_reference_latents.
    clean_reference_latents: torch.Tensor = None, 
    rgb_condition_latents: torch.Tensor = None,
    rgb_condition_mode: str = "first_frame",
    rgb_condition_scale: float = 1.0,
    **kwargs,
):
    # 1) Sliding-window path (must forward clean_reference_latents).
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            # Forward through recursive tiled call.
            clean_reference_latents=clean_reference_latents,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video_geonext, # recursive call
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    
    # 2) LongCat / S2V compatibility branches.
    if LongCatVideoTransformer3DModel is not None and isinstance(dit, LongCatVideoTransformer3DModel):
        return model_fn_longcat_video(
            dit=dit, latents=latents, timestep=timestep, context=context,
            longcat_latents=longcat_latents,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )
        
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit, latents=latents, timestep=timestep, context=context,
            audio_embeds=audio_embeds, motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents, drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    # 3) Distributed-parallel environment setup.
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    
    # If clean reference latents are provided, overwrite the leading frames.
    # latents shape: [B, C, F, H, W]
    if clean_reference_latents is not None:
        # print(f"Injecting clean reference latents with shape {clean_reference_latents.shape} into latents with shape {latents.shape}")
        # Ensure clean_reference_latents uses the correct device/dtype.
        ref = clean_reference_latents.to(device=latents.device, dtype=latents.dtype)
        num_clean_frames = ref.shape[2]
        
        # Force-replace the first N frames with clean latents.
        latents[:, :, :num_clean_frames] = ref

    if rgb_condition_mode == "repeat_add":
        if rgb_condition_latents is None:
            raise ValueError("rgb_condition_mode='repeat_add' requires rgb_condition_latents.")
        rgb_ref = rgb_condition_latents.to(device=latents.device, dtype=latents.dtype)
        if rgb_ref.shape[2] != 1:
            rgb_ref = rgb_ref[:, :, :1]
        rgb_ref = rgb_ref.repeat(1, 1, latents.shape[2], 1, 1)
        latents = latents + float(rgb_condition_scale) * rgb_ref
    elif rgb_condition_mode == "concat":
        if rgb_condition_latents is None:
            raise ValueError("rgb_condition_mode='concat' requires rgb_condition_latents.")
        rgb_ref = rgb_condition_latents.to(device=latents.device, dtype=latents.dtype)
        if rgb_ref.shape[2] != 1:
            rgb_ref = rgb_ref[:, :, :1]
        rgb_ref = rgb_ref.repeat(1, 1, latents.shape[2], 1, 1)
        latents = torch.cat([latents, float(rgb_condition_scale) * rgb_ref], dim=1)
    elif rgb_condition_mode != "first_frame":
        raise ValueError(
            f"Unsupported rgb_condition_mode: {rgb_condition_mode}. "
            "Expected 'first_frame', 'repeat_add', or 'concat'."
        )

    # =========================================================
    # 3) Reworked timestep construction logic.
    # Original logic: if dit.seperated_timestep and fuse_vae_embedding_in_latents ...
    # New logic: prioritize clean_reference_latents when present.
    
    if dit.seperated_timestep and (clean_reference_latents is not None):
        # Compute clean/noisy frame counts dynamically.
        num_clean = clean_reference_latents.shape[2]
        num_total = latents.shape[2]
        num_noisy = num_total - num_clean
        tokens_per_frame = (latents.shape[3] // 2) * (latents.shape[4] // 2)
        
        timestep = torch.concat([
            torch.zeros((num_clean, tokens_per_frame), dtype=latents.dtype, device=latents.device),
            torch.ones((num_noisy, tokens_per_frame), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()

        # Build timestep embeddings.
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        
        # Handle USP sequence parallel path.
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))

    # Fallback to the original branch (e.g., Ti2V single-frame style input).
    elif dit.seperated_timestep and fuse_vae_embedding_in_latents:
        tokens_per_frame = (latents.shape[3] // 2) * (latents.shape[4] // 2)
        timestep = torch.concat([
            torch.zeros((1, tokens_per_frame), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, tokens_per_frame), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        # Standard timestep logic.
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # 4) Motion controller.
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    
    # 5) Context embedding.
    context = dit.text_embedding(context)

    # 6) CFG merge.
    x = latents
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # 7) Image embedding (y / clip_feature).
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)
        
    # 8) Camera control.
    # Pass control_camera_latents_input to avoid patchify errors.
    x = dit.patchify(x, control_camera_latents_input)
    
    # 9) Animate adapter.
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(x, pose_latents, face_pixel_values)
    
    # 10) Flatten / patchify reshape.
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    
    # 11) Reference image path.
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    # 12) RoPE frequency construction.
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # 13) VAP logic.
    if vap is not None:
        # hidden state
        x_vap = vap_hidden_state
        x_vap = vap.patchify(x_vap)
        x_vap = rearrange(x_vap, 'b c f h w -> b (f h w) c').contiguous()
        # Timestep
        clean_timestep = torch.ones(timestep.shape, device=timestep.device).to(timestep.dtype)
        t_vap = vap.time_embedding(sinusoidal_embedding_1d(vap.freq_dim, clean_timestep))
        t_mod_vap = vap.time_projection(t_vap).unflatten(1, (6, vap.dim))

        # rope
        freqs_vap = vap.compute_freqs_mot(f,h,w).to(x.device)

        # context
        vap_clip_embedding = vap.img_emb(vap_clip_feature)
        context_vap = vap.text_embedding(context_vap)
        context_vap = torch.cat([vap_clip_embedding, context_vap], dim=1)
    
    # 14) TeaCache logic.
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    # 15) VACE logic.
    if vace_context is not None:
        vace_hints = vace(
            x, vace_context, context, t_mod, freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload
        )
    
    # 16) Transformer block loop.
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
            
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        def create_custom_forward_vap(block, vap):
            def custom_forward(*inputs):
                return vap(block, *inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            # Block logic (including VAP, checkpointing, etc.).
            if vap is not None and block_id in vap.mot_layers_mapping:
                # VAP branch.
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x, x_vap = torch.utils.checkpoint.checkpoint(
                            create_custom_forward_vap(block, vap),
                            x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x, x_vap = torch.utils.checkpoint.checkpoint(
                        create_custom_forward_vap(block, vap),
                        x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id,
                        use_reentrant=False,
                    )
                else:
                    x, x_vap = vap(block, x, context, t_mod, freqs, x_vap, context_vap, t_mod_vap, freqs_vap, block_id)
            else:
                # Standard branch.
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                elif use_gradient_checkpointing:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
                else:
                    x = block(x, context, t_mod, freqs)
            
            # VACE Injection
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
            
            # Animate Injection
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
                
        if tea_cache is not None:
            tea_cache.store(x)
            
    # 17) Head + unpatchify.
    x = dit.head(x, t)
    
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
            
    # Remove reference latents if added
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
        
    x = dit.unpatchify(x, (f, h, w))
    return x


# Backward-compatible alias.
model_fn_wan_video_three_frames = model_fn_wan_video_geonext
