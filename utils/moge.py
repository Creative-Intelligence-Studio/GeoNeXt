"""Optional MoGe metric-space alignment, imported only when requested."""

import sys
from pathlib import Path

import numpy as np

from .geometry import Camera


class MoGeAligner:
    def __init__(self, model_name="Ruicheng/moge-2-vits-normal", device="cuda"):
        try:
            import torch
        except (ImportError, OSError) as exc:
            raise ImportError(
                "PyTorch could not be loaded while initializing MoGe: %s" % exc) from exc
        try:
            source = Path(__file__).resolve().parents[1] / "third_party" / "MoGe"
            if source.is_dir() and str(source) not in sys.path:
                sys.path.insert(0, str(source))
            from moge.model import import_model_class_by_version
        except ImportError as exc:
            raise ImportError(
                "MoGe alignment requested but MoGe is not installed. "
                "Run scripts/setup_moge.sh first.") from exc
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model_class = import_model_class_by_version("v2")
        self.model = model_class.from_pretrained(model_name).to(self.device).eval()

    def infer_reference(self, rgb):
        image = np.asarray(rgb.convert("RGB"), dtype=np.float32) / 255.0
        tensor = self.torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        with self.torch.inference_mode():
            output = self.model.infer(tensor, apply_mask=False, use_fp16=self.device.type == "cuda")
        depth = output["depth"].detach().float().cpu().numpy().squeeze()
        mask = output["mask"].detach().cpu().numpy().squeeze().astype(bool)
        intrinsics = output["intrinsics"].detach().float().cpu().numpy().squeeze()
        height, width = depth.shape
        return depth, mask, Camera.from_normalized_matrix(intrinsics, width, height), intrinsics

    def align_disparity(self, disparity, rgb):
        """Fit GeoNeXt disparity affinely to MoGe metric inverse depth."""
        reference_depth, reference_mask, camera, intrinsics = self.infer_reference(rgb)
        prediction = np.asarray(disparity, dtype=np.float32).squeeze()
        if prediction.shape != reference_depth.shape:
            from PIL import Image
            prediction = np.asarray(Image.fromarray(prediction, mode="F").resize(
                (reference_depth.shape[1], reference_depth.shape[0]), Image.Resampling.BILINEAR))
        # The public output contract guarantees normalized disparity [0, 1]
        # for both Wan and SVD.
        prediction = np.clip(prediction, 0.0, 1.0)
        valid = (reference_mask & np.isfinite(reference_depth) & (reference_depth > 1e-6)
                 & np.isfinite(prediction) & (prediction > 0))
        if valid.sum() < 100:
            raise ValueError("Too few valid MoGe pixels for metric-space alignment")
        p = prediction[valid]
        q = 1.0 / reference_depth[valid]
        scale, shift = np.linalg.lstsq(
            np.column_stack((p, np.ones(p.size, dtype=np.float32))), q, rcond=None)[0]
        aligned_disparity = scale * prediction + shift
        positive = aligned_disparity[valid & (aligned_disparity > 1e-6)]
        floor = np.percentile(positive, 1) if positive.size else 1e-3
        aligned_depth = 1.0 / np.maximum(aligned_disparity, max(float(floor), 1e-6))
        return aligned_depth.astype(np.float32), reference_mask, camera, intrinsics, float(scale), float(shift)
