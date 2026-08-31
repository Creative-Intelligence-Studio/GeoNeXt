import imageio
import torch
import torch.nn.functional as F
import numpy as np
import h5py
import matplotlib
import os
from PIL import Image
from diffsynth.core.data.operators import DataProcessingOperator

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


def _read_vkitti_depth(path):
    if cv2 is not None:
        depth = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    else:
        depth = imageio.v2.imread(path)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

    if depth is None:
        raise FileNotFoundError(f"Unable to read VKITTI depth file: {path}")
    return depth.astype(np.float32) / 100.0


def _read_exr_array(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"EXR file not found: {path}")
    if os.path.getsize(path) <= 0:
        raise ValueError(f"EXR file is empty (0 bytes): {path}")

    arr = None
    if cv2 is not None:
        try:
            arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if arr is not None and arr.ndim == 3:
                # OpenCV EXR is BGR; convert to RGB for consistency.
                arr = arr[:, :, ::-1]
        except Exception:
            # Some OpenCV builds disable OpenEXR support.
            # Fall back to imageio below.
            arr = None
        if arr is None:
            # Retry once with explicit runtime flag in case environment was not set.
            try:
                os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
                arr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if arr is not None and arr.ndim == 3:
                    arr = arr[:, :, ::-1]
            except Exception:
                arr = None
    if arr is None:
        arr = imageio.v2.imread(path)
    if arr is None:
        raise FileNotFoundError(f"Unable to read EXR file: {path}")
    return np.asarray(arr, dtype=np.float32)


class LoadVideoRange(DataProcessingOperator):
    """
    输入: dict {"path": "...", "start_frame": 100, "end_frame": 200}
    输出: List[PIL.Image] (视频片段)
    """
    def __init__(self, frame_processor=lambda x: x):
        self.frame_processor = frame_processor

    def __call__(self, data: dict):
        path = data['path']
        start = data['start_frame']
        end = data['end_frame']
        
        target_count = end - start 
        
        frames = []
        reader = None
        
        try:
            reader = imageio.get_reader(path)

            reader.set_image_index(start)
            
           
            for _ in range(target_count):
                try:

                    frame = reader.get_next_data()
                    frame = Image.fromarray(frame)
                    frame = self.frame_processor(frame)
                    frames.append(frame)
                except (IndexError, RuntimeError, StopIteration):

                    break
                    
        except Exception as e:

            print(f"[Warning] Failed to read video {path} at {start}: {e}")
            
        finally:
            if reader is not None:
                reader.close()
            

        
        current_len = len(frames)
        

        if 0 < current_len < target_count:
            print(f"[Warning] Padding video {path}: {current_len}/{target_count}")
            last_frame = frames[-1]
            for _ in range(target_count - current_len):
                frames.append(last_frame)
        

        elif current_len == 0:
            print(f"[Error] Skip corrupted video: {path}")
            return None 
        return frames

class ComputeCutMask(DataProcessingOperator):
    """
    修正版：直接接收局部坐标
    输入: dict {"cuts_local": [5, 30], "num_frames": 121}
    输出: Tensor (形状 [num_frames], 切镜处为1，其余为0)
    """
    def __call__(self, data: dict):
        
        length = data['num_frames']

        cuts = data.get('cuts_local', []) 
        mask = torch.zeros(length, dtype=torch.float32)
        for idx in cuts:

            if 0 <= idx < length:
                mask[int(idx)] = 1.0
                
        return mask


def _hypersim_distance_to_depth(distance):
    int_width = 1024
    int_height = 768
    focal = 886.81

    imageplane_x = np.linspace((-0.5 * int_width) + 0.5, (0.5 * int_width) - 0.5, int_width).reshape(1, int_width).repeat(int_height, 0).astype(np.float32)[:, :, None]
    imageplane_y = np.linspace((-0.5 * int_height) + 0.5, (0.5 * int_height) - 0.5, int_height).reshape(int_height, 1).repeat(int_width, 1).astype(np.float32)[:, :, None]
    imageplane_z = np.full([int_height, int_width, 1], focal, np.float32)
    imageplane = np.concatenate([imageplane_x, imageplane_y, imageplane_z], 2)
    return distance / np.linalg.norm(imageplane, 2, 2) * focal


def _create_uv_mesh(h, w):
    y, x = np.meshgrid(
        np.arange(0, h, dtype=np.float64),
        np.arange(0, w, dtype=np.float64),
        indexing="ij",
    )
    meshgrid = np.stack((x, y))
    ones = np.ones((1, h * w), dtype=np.float64)
    xy = meshgrid.reshape(2, -1)
    return np.concatenate([xy, ones], axis=0)


def _align_normals_to_camera(normal, depth, intrinsics, h, w):
    k = np.array(
        [
            [intrinsics[0], 0, intrinsics[2]],
            [0, intrinsics[1], intrinsics[3]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    inv_k = np.linalg.inv(k)
    xy = _create_uv_mesh(h, w)
    points = np.matmul(inv_k[:3, :3], xy).reshape(3, h, w)
    points = depth * points
    points = points.transpose((1, 2, 0))
    orient_mask = np.sum(normal * points, axis=2) < 0
    normal[orient_mask] *= -1
    return normal


def _normalize_depth_to_uint8(depth, valid_mask=None, quantile_min=0.02, quantile_max=0.98):
    depth = np.asarray(depth, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (depth > 0)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)

    if valid_mask.any():
        valid_values = depth[valid_mask]
        dmin = np.quantile(valid_values, quantile_min)
        dmax = np.quantile(valid_values, quantile_max)
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            dmin = float(valid_values.min())
            dmax = float(valid_values.max())
        if dmax > dmin:
            depth = (depth - dmin) / (dmax - dmin + 1e-6)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)

    depth = np.clip(depth, 0.0, 1.0)
    return colorize_depth_map(depth, mask=valid_mask, reverse_color=True)


def colorize_depth_map(depth, mask=None, reverse_color=True):
    cm = matplotlib.colormaps["Spectral"]
    depth = np.asarray(depth, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(depth)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(depth)

    if mask.any():
        depth_min = float(depth[mask].min())
        depth_max = float(depth[mask].max())
        if np.isfinite(depth_min) and np.isfinite(depth_max) and depth_max > depth_min:
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        else:
            depth = np.zeros_like(depth, dtype=np.float32)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)

    depth = np.clip(depth, 0.0, 1.0)
    if reverse_color:
        img_colored_np = cm(1 - depth, bytes=False)[:, :, 0:3]
    else:
        img_colored_np = cm(depth, bytes=False)[:, :, 0:3]

    depth_colored = (img_colored_np * 255).astype(np.uint8)
    if mask is not None:
        masked_image = np.zeros_like(depth_colored)
        masked_image[mask] = depth_colored[mask]
        return Image.fromarray(masked_image, mode="RGB")
    return Image.fromarray(depth_colored, mode="RGB")


def _depth_to_pil_by_norm_type(depth, norm_type="truncnorm", truncnorm_min=0.02, d_max=80.0):
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.isfinite(depth) & (depth > 0)
    if valid_mask.any():
        valid_values = depth[valid_mask]
        if norm_type == "instnorm":
            dmin = float(valid_values.min())
            dmax_cur = float(valid_values.max())
        elif norm_type == "truncnorm":
            dmin = float(np.quantile(valid_values, truncnorm_min))
            dmax_cur = float(np.quantile(valid_values, 1.0 - truncnorm_min))
        elif norm_type == "perscene_norm":
            dmin = 0.0
            dmax_cur = d_max
        elif norm_type == "disparity":
            disparity = 1.0 / np.clip(depth, 1e-6, None)
            disp_values = disparity[valid_mask]
            dmin = float(disp_values.min())
            dmax_cur = float(disp_values.max())
            depth = disparity
        elif norm_type == "trunc_disparity":
            disparity = 1.0 / np.clip(depth, 1e-6, None)
            disp_values = disparity[valid_mask]
            dmin = float(np.quantile(disp_values, truncnorm_min))
            dmax_cur = float(np.quantile(disp_values, 1.0 - truncnorm_min))
            depth = disparity
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        if not np.isfinite(dmin) or not np.isfinite(dmax_cur) or dmax_cur <= dmin:
            depth = np.zeros_like(depth, dtype=np.float32)
        else:
            depth = (depth - dmin) / (dmax_cur - dmin + 1e-6)
    else:
        depth = np.zeros_like(depth, dtype=np.float32)

    depth = np.clip(depth, 0.0, 1.0)
    return colorize_depth_map(depth, mask=valid_mask, reverse_color=True)


def _normal_array_to_pil(normal):
    normal = np.asarray(normal, dtype=np.float32)
    normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    normal = np.clip(normal, -1.0, 1.0)
    normal = ((normal + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(normal, mode="RGB")


def _pil_rgb_to_signed_tensor(image: Image.Image):
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _normal_array_to_signed_tensor(normal: np.ndarray):
    normal = np.asarray(normal, dtype=np.float32)
    normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
    normal = np.clip(normal, -1.0, 1.0)
    return torch.from_numpy(normal).permute(2, 0, 1).contiguous()


def _depth_to_signed_tensor_by_norm_type(depth, norm_type="truncnorm", truncnorm_min=0.02, d_max=80.0):
    depth = np.asarray(depth, dtype=np.float32)
    valid_mask = np.isfinite(depth) & (depth > 0)
    if d_max is not None:
        valid_mask = valid_mask & (depth < float(d_max))

    depth_norm = np.zeros_like(depth, dtype=np.float32)
    if valid_mask.any():
        valid_values = depth[valid_mask]
        if norm_type == "instnorm":
            dmin = float(valid_values.min())
            dmax_cur = float(valid_values.max())
            values = depth
        elif norm_type == "truncnorm":
            dmin = float(np.quantile(valid_values, truncnorm_min))
            dmax_cur = float(np.quantile(valid_values, 1.0 - truncnorm_min))
            values = depth
        elif norm_type == "perscene_norm":
            dmin = 0.0
            dmax_cur = float(d_max)
            values = depth
        elif norm_type == "disparity":
            values = 1.0 / np.clip(depth, 1e-6, None)
            disp_values = values[valid_mask]
            dmin = float(disp_values.min())
            dmax_cur = float(disp_values.max())
        elif norm_type == "trunc_disparity":
            values = 1.0 / np.clip(depth, 1e-6, None)
            disp_values = values[valid_mask]
            dmin = float(np.quantile(disp_values, truncnorm_min))
            dmax_cur = float(np.quantile(disp_values, 1.0 - truncnorm_min))
        else:
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        if np.isfinite(dmin) and np.isfinite(dmax_cur) and dmax_cur > dmin:
            depth_norm = ((values - dmin) / (dmax_cur - dmin + 1e-6) - 0.5) * 2.0

    depth_norm = np.clip(depth_norm, -1.0, 1.0).astype(np.float32)
    depth_chw = np.repeat(depth_norm[None, :, :], 3, axis=0)
    return torch.from_numpy(depth_chw).contiguous()


class ResizeShortestEdge(DataProcessingOperator):
    def __init__(self, resolution, interpolation=Image.BILINEAR):
        self.resolution = resolution
        self.interpolation = interpolation

    def __call__(self, image: Image.Image):
        width, height = image.size
        if height > width:
            new_width = self.resolution
            new_height = int(self.resolution * height / width)
        else:
            new_height = self.resolution
            new_width = int(self.resolution * width / height)
        return image.resize((new_width, new_height), self.interpolation)


class VKITTICropAndResize(DataProcessingOperator):
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216

    def __init__(self, interpolation=Image.BILINEAR):
        self.interpolation = interpolation

    def _resize_if_needed(self, image: Image.Image):
        current_width, current_height = image.size
        if current_height < self.KB_CROP_HEIGHT or current_width < self.KB_CROP_WIDTH:
            scaling_factor = max(self.KB_CROP_HEIGHT / current_height, self.KB_CROP_WIDTH / current_width)
            new_width = int(current_width * scaling_factor)
            new_height = int(current_height * scaling_factor)
            image = image.resize((new_width, new_height), self.interpolation)
        return image

    def __call__(self, image: Image.Image):
        image = self._resize_if_needed(image)
        top = int(image.height - self.KB_CROP_HEIGHT)
        left = int((image.width - self.KB_CROP_WIDTH) / 2)
        return image.crop((left, top, left + self.KB_CROP_WIDTH, top + self.KB_CROP_HEIGHT))


def _vkitti_resize_crop_depth(depth_tensor):
    crop_height = VKITTICropAndResize.KB_CROP_HEIGHT
    crop_width = VKITTICropAndResize.KB_CROP_WIDTH
    _, _, current_height, current_width = depth_tensor.shape
    if current_height < crop_height or current_width < crop_width:
        scaling_factor = max(crop_height / current_height, crop_width / current_width)
        new_height = int(current_height * scaling_factor)
        new_width = int(current_width * scaling_factor)
        depth_tensor = F.interpolate(depth_tensor, size=(new_height, new_width), mode="nearest")
    top = int(depth_tensor.shape[-2] - crop_height)
    left = int((depth_tensor.shape[-1] - crop_width) / 2)
    return depth_tensor[:, :, top:top + crop_height, left:left + crop_width]


class LoadDepthMaskItem(DataProcessingOperator):
    def __init__(self, mask_kind="valid"):
        self.mask_kind = mask_kind
        self.vkitti_d_min = 1e-5
        self.vkitti_d_max = 80.0

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(f"Expected a modality dict, got {type(data)}")

        source = data.get("source")
        path = data["path"]

        if source == "hypersim":
            return None

        if source == "interiorverse":
            mask_arr = _read_exr_array(path)
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[:, :, 0]
            mask = torch.from_numpy(mask_arr.astype(np.float32)).unsqueeze(0)
            valid = torch.isfinite(mask) & (mask > 0.5)
            if self.mask_kind == "valid":
                return valid.bool()
            if self.mask_kind == "sky":
                # InteriorVerse mask does not provide explicit sky labels.
                return torch.zeros_like(valid, dtype=torch.bool)
            raise ValueError(f"Unsupported mask kind: {self.mask_kind}")

        if source != "vkitti":
            raise ValueError(f"Unsupported mask source: {source}")

        depth = _read_vkitti_depth(path)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = _vkitti_resize_crop_depth(depth).squeeze(0)

        if self.mask_kind == "valid":
            mask = torch.logical_and(depth > self.vkitti_d_min, depth < self.vkitti_d_max)
        elif self.mask_kind == "sky":
            mask = torch.logical_and(depth > self.vkitti_d_min, depth >= self.vkitti_d_max)
        else:
            raise ValueError(f"Unsupported mask kind: {self.mask_kind}")

        return mask.bool()


class LoadRGBDepthNormalItem(DataProcessingOperator):
    """
    Load a modality-tagged item into a PIL image while preserving the downstream
    sample format expected by the current three-image training pipeline.
    """
    def __init__(self, frame_processor_map=None, norm_type="truncnorm", truncnorm_min=0.02, align_cam_normal=False, output_tensor=True):
        self.frame_processor_map = {} if frame_processor_map is None else frame_processor_map
        self.norm_type = norm_type
        self.truncnorm_min = truncnorm_min
        self.align_cam_normal = align_cam_normal
        self.output_tensor = output_tensor
        self.hypersim_resolution = 576
        self.vkitti_d_max = 80.0
        # Keep InteriorVerse depth-range cap aligned with Hypersim by default.
        self.hypersim_d_max = 65.0
        self.interiorverse_d_max = self.hypersim_d_max
        processor = self.frame_processor_map.get(("hypersim", "depth"))
        if isinstance(processor, ResizeShortestEdge):
            self.hypersim_resolution = int(processor.resolution)

    def _postprocess(self, image, data):
        source = data.get("source")
        kind = data.get("kind", "rgb")
        processor = self.frame_processor_map.get((source, kind))
        if processor is None:
            processor = self.frame_processor_map.get((source, None))
        if processor is None:
            processor = self.frame_processor_map.get((None, kind))
        if processor is None:
            processor = self.frame_processor_map.get((None, None))
        return processor(image) if processor is not None else image

    def _resize_depth(self, depth, source):
        depth = np.asarray(depth, dtype=np.float32)
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        if source == "vkitti":
            depth_t = _vkitti_resize_crop_depth(depth_t)
        elif source == "hypersim":
            h, w = depth_t.shape[-2:]
            if h > w:
                new_w = self.hypersim_resolution
                new_h = int(self.hypersim_resolution * h / w)
            else:
                new_h = self.hypersim_resolution
                new_w = int(self.hypersim_resolution * w / h)
            depth_t = F.interpolate(depth_t, size=(new_h, new_w), mode="nearest")
        else:
            raise ValueError(f"Unsupported depth source: {source}")
        return depth_t.squeeze(0).squeeze(0).cpu().numpy()

    def _load_vkitti_depth(self, path):
        return _read_vkitti_depth(path)

    def _load_hypersim_depth(self, path):
        with h5py.File(path, "r") as handle:
            distance = np.array(handle["dataset"], dtype=np.float32)
        return _hypersim_distance_to_depth(distance)

    def _load_hypersim_normal(self, path):
        with h5py.File(path, "r") as handle:
            normal = np.array(handle["dataset"], dtype=np.float32)
        # Match the Lotus convention that flips the camera-space x axis.
        normal[:, :, 0] *= -1.0
        if self.align_cam_normal:
            h, w = normal.shape[:2]
            depth_path = path.replace("normal_cam.hdf5", "depth_meters.hdf5")
            with h5py.File(depth_path, "r") as handle:
                distance = np.array(handle["dataset"], dtype=np.float32)
            depth = _hypersim_distance_to_depth(distance)
            normal = _align_normals_to_camera(normal, depth, [886.81, 886.81, w / 2, h / 2], h, w)
        return normal

    def _load_interiorverse_depth(self, path):
        depth = _read_exr_array(path)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth = depth.astype(np.float32)
        # InteriorVerse depth is commonly stored in millimeters.
        # Auto-detect large-scale values and convert to meters.
        finite = depth[np.isfinite(depth)]
        if finite.size > 0:
            p99 = float(np.percentile(finite, 99.0))
            if p99 > 300.0:
                depth = depth / 1000.0
        return depth

    def _load_interiorverse_rgb(self, path):
        rgb = _read_exr_array(path)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        if rgb.shape[-1] > 3:
            rgb = rgb[:, :, :3]
        rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        # InteriorVerse stores linear HDR. Clamp to SDR range for this training path.
        rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    def _load_interiorverse_normal(self, path):
        normal = _read_exr_array(path)
        if normal.ndim == 2:
            normal = np.stack([normal, normal, normal], axis=-1)
        if normal.shape[-1] > 3:
            normal = normal[:, :, :3]
        normal = np.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        # Support both [0,1] and [-1,1] conventions.
        if normal.min() >= -1e-4 and normal.max() <= 1.0 + 1e-4:
            normal = normal * 2.0 - 1.0
        normal = np.clip(normal, -1.0, 1.0)
        return normal

    def __call__(self, data):
        if not isinstance(data, dict):
            raise TypeError(f"Expected a modality dict, got {type(data)}")

        path = data["path"]
        kind = data.get("kind", "rgb")
        source = data.get("source")

        if kind == "rgb":
            image = Image.open(path).convert("RGB")
            image = self._postprocess(image, data)
            return _pil_rgb_to_signed_tensor(image) if self.output_tensor else image
        elif kind == "depth":
            if source == "vkitti":
                depth = self._load_vkitti_depth(path)
                d_max = self.vkitti_d_max
            elif source == "hypersim":
                depth = self._load_hypersim_depth(path)
                d_max = self.hypersim_d_max
            elif source == "interiorverse":
                depth = self._load_interiorverse_depth(path)
                d_max = self.interiorverse_d_max
            else:
                raise ValueError(f"Unsupported depth source: {source}")
            if source in ("vkitti", "hypersim"):
                depth = self._resize_depth(depth, source)
            else:
                image = Image.fromarray(depth.astype(np.float32), mode="F")
                image = self._postprocess(image, data)
                depth = np.asarray(image, dtype=np.float32)
            if self.output_tensor:
                return _depth_to_signed_tensor_by_norm_type(
                    depth,
                    norm_type=self.norm_type,
                    truncnorm_min=self.truncnorm_min,
                    d_max=d_max,
                )
            image = _depth_to_pil_by_norm_type(
                depth,
                norm_type=self.norm_type,
                truncnorm_min=self.truncnorm_min,
                d_max=d_max,
            )
            return image
        elif kind == "normal":
            if source == "vkitti":
                image = Image.open(path).convert("RGB")
                image = self._postprocess(image, data)
                return _pil_rgb_to_signed_tensor(image) if self.output_tensor else image
            elif source == "hypersim":
                normal = self._load_hypersim_normal(path)
                if self.output_tensor:
                    normal_t = torch.from_numpy(normal.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
                    h, w = normal_t.shape[-2:]
                    if h > w:
                        new_w = self.hypersim_resolution
                        new_h = int(self.hypersim_resolution * h / w)
                    else:
                        new_h = self.hypersim_resolution
                        new_w = int(self.hypersim_resolution * w / h)
                    normal_t = F.interpolate(normal_t, size=(new_h, new_w), mode="nearest").squeeze(0)
                    normal_t = torch.clamp(normal_t, -1.0, 1.0)
                    return normal_t.contiguous()
                image = _normal_array_to_pil(normal)
                image = self._postprocess(image, data)
                return image
            elif source == "interiorverse":
                normal = self._load_interiorverse_normal(path)
                if self.output_tensor:
                    normal_t = torch.from_numpy(normal.astype(np.float32)).permute(2, 0, 1)
                    img = Image.fromarray(((np.clip(normal, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8), mode="RGB")
                    img = self._postprocess(img, data)
                    return _pil_rgb_to_signed_tensor(img)
                image = _normal_array_to_pil(normal)
                image = self._postprocess(image, data)
                return image
            else:
                raise ValueError(f"Unsupported normal source: {source}")
        elif kind == "rgb_exr":
            if source != "interiorverse":
                raise ValueError(f"Unsupported rgb_exr source: {source}")
            rgb = self._load_interiorverse_rgb(path)
            image = Image.fromarray((rgb * 255.0).astype(np.uint8), mode="RGB")
            image = self._postprocess(image, data)
            return _pil_rgb_to_signed_tensor(image) if self.output_tensor else image
        else:
            raise ValueError(f"Unsupported modality kind: {kind}")
