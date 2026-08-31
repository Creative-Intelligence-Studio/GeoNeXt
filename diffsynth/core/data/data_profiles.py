import json, os, random
import numpy as np
from PIL import Image
from diffsynth.core.data.operators import ImageCropAndResize,SequencialProcess,LoadImage
from diffsynth.core.data.custom_operators import (
    LoadVideoRange,
    ComputeCutMask,
    LoadRGBDepthNormalItem,
    LoadDepthMaskItem,
    ResizeShortestEdge,
    VKITTICropAndResize,
)
from diffsynth.core.data.custom_units import WanVideoUnit_GeoNeXtFused
from diffsynth.core.data.custom_model_fn import model_fn_wan_video_geonext
class BaseDataProfile:
    def __init__(self, args):
        self.args = args
    def load_and_transform(self, path): raise NotImplementedError
    def get_operator_map(self): return {}
    def get_data_keys(self): return []
    def get_extra_inputs(self): return []
    def configure_pipeline(self, pipe):pass
    def postprocess_sample(self, data): return data


import torch
import torch.nn as nn

class DepthNormalProfile(BaseDataProfile):
    def __init__(self, args):
        super().__init__(args)
        self.fixed_sample_rgb_path = str(getattr(args, "fixed_sample_rgb_path", "") or "").strip()
        self.random_flip = bool(getattr(args, "random_flip", False))
        self.interiorverse_use_mask = bool(getattr(args, "interiorverse_use_mask", True))
        self.interiorverse_cache_index = bool(getattr(args, "interiorverse_cache_index", True))
        self.refresh_interiorverse_index = bool(getattr(args, "refresh_interiorverse_index", False))
        self.interiorverse_index_cache_path = str(getattr(args, "interiorverse_index_cache_path", "") or "").strip()
        self.loss_mask_mode = str(getattr(args, "loss_mask_mode", "depth_only"))
        target_modalities = str(getattr(args, "target_modalities", "depth,normal"))
        self.target_modalities = [m.strip() for m in target_modalities.split(",") if m.strip()]
        if len(self.target_modalities) == 0:
            self.target_modalities = ["depth", "normal"]
        valid_modalities = {"depth", "normal"}
        invalid = [m for m in self.target_modalities if m not in valid_modalities]
        if invalid:
            raise ValueError(
                f"Unsupported target_modalities={invalid}. Allowed: {sorted(valid_modalities)}"
            )
        # Keep depth as first predicted slot when present so legacy depth-only mask
        # semantics still make sense.
        if "depth" in self.target_modalities:
            self.target_modalities = ["depth"] + [m for m in self.target_modalities if m != "depth"]
        self.num_condition_frames = int(getattr(args, "num_condition_frames", 1))
        self.modalities = ["rgb"] + list(self.target_modalities)
        self.supervise_modalities_by_source = {
            "hypersim": self._parse_source_supervise_modalities(getattr(args, "supervise_modalities_hypersim", "")),
            "vkitti": self._parse_source_supervise_modalities(getattr(args, "supervise_modalities_vkitti", "")),
            "interiorverse": self._parse_source_supervise_modalities(getattr(args, "supervise_modalities_interiorverse", "")),
        }

    def _parse_source_supervise_modalities(self, text):
        text = str(text or "").strip()
        if len(text) == 0:
            return set(self.target_modalities)
        items = {m.strip() for m in text.split(",") if m.strip()}
        return {m for m in items if m in set(self.target_modalities)}

    def _default_interiorverse_cache_path(self, root_dir, fov_mode):
        safe_modalities = "_".join(self.target_modalities)
        norm_root = os.path.normpath(root_dir)
        # Preferred layout for current cluster dataset path:
        #   /.../datasets/interiorVerse/interiorverse/interverse
        # Save cache at:
        #   /.../datasets/interiorVerse/.index_cache
        if norm_root.endswith(os.path.join("interiorverse", "interverse")):
            cache_base = os.path.dirname(os.path.dirname(norm_root))
            cache_dir = os.path.join(cache_base, ".index_cache")
        else:
            cache_dir = os.path.join(root_dir, ".index_cache")
        return os.path.join(cache_dir, f"interiorverse_index_{fov_mode}_{safe_modalities}.json")

    def _mix_sources(self, source_to_data, mix_seed):
        available = {k: v for k, v in source_to_data.items() if len(v) > 0}
        if len(available) == 0:
            return []
        if len(available) == 1:
            only_source = next(iter(available.keys()))
            print(f"[DepthNormalProfile] Only {only_source} is available; sampling ratio options are ignored.")
            return list(next(iter(available.values())))

        p_h = float(getattr(self.args, "p_hypersim", 0.5))
        p_i = float(getattr(self.args, "p_interiorverse", 0.0))
        p_h = max(0.0, p_h)
        p_i = max(0.0, p_i)
        p_v = max(0.0, 1.0 - p_h - p_i)
        raw_weights = {
            "hypersim": p_h,
            "vkitti": p_v,
            "interiorverse": p_i,
        }
        weights = {k: raw_weights.get(k, 0.0) for k in available.keys()}
        total_w = sum(weights.values())
        if total_w <= 0:
            uniform = 1.0 / len(available)
            weights = {k: uniform for k in available.keys()}
        else:
            weights = {k: v / total_w for k, v in weights.items()}

        total_len = sum(len(v) for v in available.values())
        counts = {k: int(round(total_len * w)) for k, w in weights.items()}
        # Ensure every available source contributes at least one sample.
        for k in list(counts.keys()):
            counts[k] = max(1, counts[k])
        # Fix total count drift.
        drift = sum(counts.values()) - total_len
        keys_sorted = sorted(counts.keys(), key=lambda k: counts[k], reverse=(drift > 0))
        idx = 0
        while drift != 0 and len(keys_sorted) > 0:
            key = keys_sorted[idx % len(keys_sorted)]
            if drift > 0 and counts[key] > 1:
                counts[key] -= 1
                drift -= 1
            elif drift < 0:
                counts[key] += 1
                drift += 1
            idx += 1
            if idx > 100000:
                break

        rng = random.Random(mix_seed)
        mixed = []
        for key, data in available.items():
            n = counts[key]
            if n <= len(data):
                sampled = rng.sample(data, n)
            else:
                sampled = [rng.choice(data) for _ in range(n)]
            mixed.extend(sampled)
        rng.shuffle(mixed)
        print(
            "[DepthNormalProfile] Mixed sources with ratios="
            + ", ".join([f"{k}:{weights[k]:.3f}" for k in sorted(weights.keys())])
            + " -> counts="
            + ", ".join([f"{k}:{counts[k]}" for k in sorted(counts.keys())])
            + f", total={len(mixed)}"
        )
        return mixed

    def _expand_patch_embedding_for_concat(self, dit, extra_in_channels):
        old_conv = dit.patch_embedding
        new_in_channels = old_conv.in_channels + int(extra_in_channels)
        if old_conv.in_channels == new_in_channels:
            return
        if old_conv.in_channels > new_in_channels:
            raise ValueError(
                f"Cannot shrink patch_embedding from {old_conv.in_channels} to {new_in_channels} channels."
            )

        new_conv = nn.Conv3d(
            new_in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=old_conv.bias is not None,
            padding_mode=old_conv.padding_mode,
        ).to(device=old_conv.weight.device, dtype=old_conv.weight.dtype)
        with torch.no_grad():
            new_conv.weight.zero_()
            new_conv.weight[:, :old_conv.in_channels].copy_(old_conv.weight)
            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        dit.patch_embedding = new_conv
        dit.in_dim = new_in_channels
        print(
            f"[Config] Expanded DiT patch_embedding input channels "
            f"{old_conv.in_channels} -> {new_in_channels} for rgb_condition_mode=concat."
        )

    def _build_entry(self, source, modality_paths, valid_mask_item=None, sky_mask_item=None):
        rgb_kind = "rgb_exr" if source == "interiorverse" else "rgb"
        video = [{"path": modality_paths["rgb"], "kind": rgb_kind, "source": source}]
        target_kinds_in_video = []
        for kind in self.target_modalities:
            path = modality_paths.get(kind)
            if path is None:
                continue
            video.append({"path": path, "kind": kind, "source": source})
            target_kinds_in_video.append(kind)
        if len(video) < 2:
            return None
        first_target = video[min(1, len(video) - 1)].copy()
        if valid_mask_item is None and source in ("vkitti", "hypersim"):
            valid_mask_item = first_target.copy()
        if sky_mask_item is None and source in ("vkitti", "hypersim"):
            sky_mask_item = first_target.copy()
        entry = {
            "video": video,
            "prompt": "",
            "first_frame": video[0].copy(),
            "second_frame": first_target,
            "sample_rgb_path": modality_paths["rgb"],
            "sample_source": source,
        }
        supervise_set = self.supervise_modalities_by_source.get(source, set(self.target_modalities))
        target_frame_loss_mask = [1.0 if kind in supervise_set else 0.0 for kind in target_kinds_in_video]
        entry["target_frame_loss_mask"] = torch.tensor(target_frame_loss_mask, dtype=torch.float32)
        if valid_mask_item is not None:
            entry["valid_mask"] = valid_mask_item
        if sky_mask_item is not None:
            entry["sky_mask"] = sky_mask_item
        return entry

    def _load_vkitti(self, root_dir):
        flattened_data = []
        scenes = ["02", "06", "18", "20"]
        conditions = [
            "15-deg-left", "15-deg-right", "30-deg-left", "30-deg-right", "clone",
            "fog", "morning", "overcast", "rain", "sunset"
        ]
        cameras = ["0", "1"]

        for scene in scenes:
            for condition in conditions:
                for camera in cameras:
                    image_dir = os.path.join(root_dir, f"Scene{scene}/{condition}/frames/rgb/Camera_{camera}")
                    if not os.path.isdir(image_dir):
                        continue
                    for image_name in sorted(os.listdir(image_dir)):
                        rgb_path = os.path.join(image_dir, image_name)
                        depth_path = rgb_path.replace("/rgb/", "/depth/").replace("rgb_", "depth_").replace(".jpg", ".png")
                        normal_path = rgb_path.replace("/rgb/", "/normal/").replace("rgb_", "normal_").replace(".jpg", ".png")
                        modality_paths = {"rgb": rgb_path}
                        if "depth" in self.target_modalities:
                            modality_paths["depth"] = depth_path
                        if "normal" in self.target_modalities:
                            modality_paths["normal"] = normal_path
                        missing = [k for k in self.target_modalities if k not in modality_paths or not os.path.exists(modality_paths[k])]
                        if len(missing) > 0 and any(m in ("depth", "normal") for m in missing):
                            continue
                        entry = self._build_entry("vkitti", modality_paths)
                        if entry is not None:
                            flattened_data.append(entry)
        print(f"[DepthNormalProfile] Loaded {len(flattened_data)} VKITTI samples.")
        return flattened_data

    def _load_hypersim(self, root_dir):
        flattened_data = []
        split = getattr(self.args, "depthnormal_hypersim_split", "train")
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            print(f"[DepthNormalProfile] Hypersim split directory not found: {split_dir}")
            return flattened_data

        for current_root, _, files in os.walk(split_dir):
            for file_name in sorted(files):
                if not file_name.endswith("tonemap.jpg"):
                    continue
                rgb_path = os.path.join(current_root, file_name)
                modality_paths = {"rgb": rgb_path}
                if "depth" in self.target_modalities:
                    modality_paths["depth"] = rgb_path.replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "depth_meters.hdf5")
                if "normal" in self.target_modalities:
                    modality_paths["normal"] = rgb_path.replace("final_preview", "geometry_hdf5").replace("tonemap.jpg", "normal_cam.hdf5")
                missing = [k for k in self.target_modalities if k not in modality_paths or not os.path.exists(modality_paths[k])]
                if len(missing) > 0 and any(m in ("depth", "normal") for m in missing):
                    continue
                entry = self._build_entry("hypersim", modality_paths)
                if entry is not None:
                    flattened_data.append(entry)
        print(f"[DepthNormalProfile] Loaded {len(flattened_data)} Hypersim samples.")
        return flattened_data

    def _load_interiorverse(self, root_dir):
        def _is_nonempty_file(path):
            return os.path.isfile(path) and (os.path.getsize(path) > 0)

        fov_mode = str(getattr(self.args, "interiorverse_fov", "85")).lower()
        cache_path = self.interiorverse_index_cache_path or self._default_interiorverse_cache_path(root_dir, fov_mode)
        use_cache = self.interiorverse_cache_index and not self.refresh_interiorverse_index
        if use_cache and os.path.isfile(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                if isinstance(cached, list) and len(cached) > 0:
                    print(f"[DepthNormalProfile] Loaded InteriorVerse index cache: {cache_path} ({len(cached)} samples)")
                    return cached
            except Exception as exc:
                print(f"[DepthNormalProfile] Failed to read InteriorVerse index cache ({cache_path}): {exc}")

        flattened_data = []
        subset_dirs = []
        if fov_mode in ("85", "both", "all"):
            subset_dirs.append(os.path.join(root_dir, "85"))
        if fov_mode in ("120", "both", "all"):
            subset_dirs.append(os.path.join(root_dir, "120_part"))

        for subset in subset_dirs:
            if not os.path.isdir(subset):
                print(f"[DepthNormalProfile] InteriorVerse subset not found: {subset}")
                continue
            for scene_name in sorted(os.listdir(subset)):
                scene_dir = os.path.join(subset, scene_name)
                if not os.path.isdir(scene_dir):
                    continue
                rgb_names = [n for n in os.listdir(scene_dir) if n.endswith("_im.exr")]
                for rgb_name in sorted(rgb_names):
                    stem = rgb_name[:-7]
                    modality_paths = {"rgb": os.path.join(scene_dir, rgb_name)}
                    if "depth" in self.target_modalities:
                        modality_paths["depth"] = os.path.join(scene_dir, f"{stem}_depth.exr")
                    if "normal" in self.target_modalities:
                        modality_paths["normal"] = os.path.join(scene_dir, f"{stem}_normal.exr")
                    required_keys = ["rgb"] + list(self.target_modalities)
                    missing = [
                        k
                        for k in required_keys
                        if (k not in modality_paths) or (not _is_nonempty_file(modality_paths[k]))
                    ]
                    if len(missing) > 0:
                        continue

                    valid_mask_item = None
                    sky_mask_item = None
                    if self.interiorverse_use_mask:
                        mask_path = os.path.join(scene_dir, f"{stem}_mask.exr")
                        if _is_nonempty_file(mask_path):
                            valid_mask_item = {"path": mask_path, "kind": "mask", "source": "interiorverse"}
                            sky_mask_item = {"path": mask_path, "kind": "mask", "source": "interiorverse"}
                    entry = self._build_entry(
                        "interiorverse",
                        modality_paths,
                        valid_mask_item=valid_mask_item,
                        sky_mask_item=sky_mask_item,
                    )
                    if entry is not None:
                        flattened_data.append(entry)

        if self.interiorverse_cache_index and len(flattened_data) > 0:
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "w", encoding="utf-8") as handle:
                    json.dump(flattened_data, handle)
                print(f"[DepthNormalProfile] Saved InteriorVerse index cache: {cache_path} ({len(flattened_data)} samples)")
            except Exception as exc:
                print(f"[DepthNormalProfile] Failed to write InteriorVerse index cache ({cache_path}): {exc}")

        print(f"[DepthNormalProfile] Loaded {len(flattened_data)} InteriorVerse samples.")
        return flattened_data

    def load_and_transform(self, metadata_path):
        flattened_data = []
        hypersim_root = getattr(self.args, "hypersim_root", None)
        vkitti_root = getattr(self.args, "vkitti_root", None)
        interiorverse_root = getattr(self.args, "interiorverse_root", None)
        mix_seed = int(getattr(self.args, "mix_seed", 42))

        hypersim_data = []
        vkitti_data = []
        interiorverse_data = []
        if hypersim_root:
            hypersim_data = self._load_hypersim(hypersim_root)
        if vkitti_root:
            vkitti_data = self._load_vkitti(vkitti_root)
        if interiorverse_root:
            interiorverse_data = self._load_interiorverse(interiorverse_root)

        if len(hypersim_data) == 0 and len(vkitti_data) == 0 and len(interiorverse_data) == 0:
            raise ValueError(
                "DepthNormalProfile found no samples. Set --hypersim_root and/or --vkitti_root and/or --interiorverse_root "
                "to valid dataset directories."
            )

        # Hard lock to one exact source sample if requested.
        if len(self.fixed_sample_rgb_path) > 0:
            all_entries = hypersim_data + vkitti_data + interiorverse_data
            target = os.path.normpath(self.fixed_sample_rgb_path)
            matched = None
            for entry in all_entries:
                try:
                    first_item = entry["video"][0]
                    rgb_path = first_item["path"] if isinstance(first_item, dict) else first_item
                except Exception:
                    continue
                if os.path.normpath(str(rgb_path)) == target:
                    matched = entry
                    break
            if matched is None:
                raise ValueError(
                    f"fixed_sample_rgb_path not found in loaded data: {self.fixed_sample_rgb_path}"
                )
            print(f"[DepthNormalProfile] Using fixed_sample_rgb_path: {self.fixed_sample_rgb_path}")
            return [matched]

        flattened_data = self._mix_sources(
            {
                "hypersim": hypersim_data,
                "vkitti": vkitti_data,
                "interiorverse": interiorverse_data,
            },
            mix_seed=mix_seed,
        )

        print(f"[DepthNormalProfile] Total mixed samples: {len(flattened_data)}")
        max_samples = int(getattr(self.args, "depthnormal_max_samples", 0) or 0)
        if max_samples > 0 and len(flattened_data) > max_samples:
            flattened_data = flattened_data[:max_samples]
            print(f"[DepthNormalProfile] Truncated to depthnormal_max_samples={max_samples}.")
        return flattened_data

    def get_operator_map(self):
        hypersim_resolution = getattr(self.args, "resolution_hypersim", 576)
        interiorverse_resolution = getattr(self.args, "resolution_interiorverse", 576)
        processor_map = {
            ("hypersim", "rgb"): ResizeShortestEdge(hypersim_resolution, interpolation=Image.BILINEAR),
            ("hypersim", "depth"): ResizeShortestEdge(hypersim_resolution, interpolation=Image.NEAREST),
            ("hypersim", "normal"): ResizeShortestEdge(hypersim_resolution, interpolation=Image.NEAREST),
            ("vkitti", "rgb"): VKITTICropAndResize(interpolation=Image.BILINEAR),
            ("vkitti", "depth"): VKITTICropAndResize(interpolation=Image.NEAREST),
            ("vkitti", "normal"): VKITTICropAndResize(interpolation=Image.NEAREST),
            ("interiorverse", "rgb_exr"): ResizeShortestEdge(interiorverse_resolution, interpolation=Image.BILINEAR),
            ("interiorverse", "depth"): ResizeShortestEdge(interiorverse_resolution, interpolation=Image.NEAREST),
            ("interiorverse", "normal"): ResizeShortestEdge(interiorverse_resolution, interpolation=Image.NEAREST),
        }
        load_item = LoadRGBDepthNormalItem(
            frame_processor_map=processor_map,
            norm_type=getattr(self.args, "norm_type", "trunc_disparity"),
            truncnorm_min=getattr(self.args, "truncnorm_min", 0.02),
            align_cam_normal=bool(getattr(self.args, "align_cam_normal", False)),
            output_tensor=True,
        )
        valid_mask_loader = LoadDepthMaskItem(mask_kind="valid")
        sky_mask_loader = LoadDepthMaskItem(mask_kind="sky")
        return {
            "video": SequencialProcess(load_item),
            "first_frame": load_item,
            "second_frame": load_item,
            "valid_mask": valid_mask_loader,
            "sky_mask": sky_mask_loader,
        }

    def get_data_keys(self):
        return ["video", "first_frame", "second_frame", "valid_mask", "sky_mask"]

    def get_extra_inputs(self):
        return ["first_frame", "second_frame", "valid_mask", "sky_mask", "target_frame_loss_mask"]

    def postprocess_sample(self, data):
        if not self.random_flip or random.random() <= 0.5:
            return data

        if "video" in data and len(data["video"]) >= 1:
            for frame_idx, frame in enumerate(data["video"]):
                if isinstance(frame, Image.Image):
                    flipped = frame.transpose(Image.FLIP_LEFT_RIGHT)
                    data["video"][frame_idx] = flipped
                elif isinstance(frame, torch.Tensor):
                    flipped = torch.flip(frame, dims=[-1]).clone()
                    # For target normal frame, invert x after horizontal flip.
                    if frame_idx > 0:
                        tgt_idx = frame_idx - 1
                        if tgt_idx < len(self.target_modalities) and self.target_modalities[tgt_idx] == "normal":
                            flipped[0] = -flipped[0]
                    data["video"][frame_idx] = flipped

        if "first_frame" in data and isinstance(data["first_frame"], Image.Image):
            data["first_frame"] = data["first_frame"].transpose(Image.FLIP_LEFT_RIGHT)
        elif "first_frame" in data and isinstance(data["first_frame"], torch.Tensor):
            data["first_frame"] = torch.flip(data["first_frame"], dims=[-1])

        if "second_frame" in data and isinstance(data["second_frame"], Image.Image):
            data["second_frame"] = data["second_frame"].transpose(Image.FLIP_LEFT_RIGHT)
        elif "second_frame" in data and isinstance(data["second_frame"], torch.Tensor):
            data["second_frame"] = torch.flip(data["second_frame"], dims=[-1])

        if "valid_mask" in data and isinstance(data["valid_mask"], torch.Tensor):
            data["valid_mask"] = torch.flip(data["valid_mask"], dims=[-1])
        if "sky_mask" in data and isinstance(data["sky_mask"], torch.Tensor):
            data["sky_mask"] = torch.flip(data["sky_mask"], dims=[-1])

        return data

    def configure_pipeline(self, pipe):
        print(
            "[Config] Configuring GeoNeXt RGB->depth/normal fused inference "
            f"(targets={self.target_modalities}, num_condition_frames={self.num_condition_frames})."
        )

        current_in_dim = pipe.dit.in_dim
        if current_in_dim not in [16, 48]:
            raise ValueError(
                f"[Config] Unsupported GeoNeXt base-model input dimension: {current_in_dim}. "
                "Expected a 16-channel T2V or 48-channel Ti2V model."
            )

        pipe.dit.seperated_timestep = True
        units_to_remove = [
            "WanVideoUnit_ShapeChecker",
            "WanVideoUnit_InputVideoEmbedder",
            "WanVideoUnit_ImageEmbedderVAE",
            "WanVideoUnit_ImageEmbedderFused",
        ]
        pipe.units = [u for u in pipe.units if u.__class__.__name__ not in units_to_remove]

        insert_index = 0
        for i, unit in enumerate(pipe.units):
            if unit.__class__.__name__ == "WanVideoUnit_NoiseInitializer":
                insert_index = i + 1
                break

        rgb_condition_mode = getattr(self.args, "rgb_condition_mode", "first_frame")
        rgb_condition_scale = float(getattr(self.args, "rgb_condition_scale", 1.0))
        if rgb_condition_mode == "concat":
            self._expand_patch_embedding_for_concat(pipe.dit, extra_in_channels=16)
        pipe.units.insert(
            insert_index,
            WanVideoUnit_GeoNeXtFused(
                num_condition_frames=self.num_condition_frames,
                rgb_condition_mode=rgb_condition_mode,
                rgb_condition_scale=rgb_condition_scale,
                loss_mask_mode=self.loss_mask_mode,
            ),
        )
        pipe.model_fn = model_fn_wan_video_geonext
        print(
            "[Config] Switched to model_fn_wan_video_geonext "
            f"(rgb_condition_mode={rgb_condition_mode}, rgb_condition_scale={rgb_condition_scale}, "
            f"loss_mask_mode={self.loss_mask_mode})."
        )
