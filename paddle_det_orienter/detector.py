import logging
import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from .schema import TextDetectionResult
from .config import PaddleDetectorConfig
from .postprocess import db_postprocess

log = logging.getLogger(__name__)

# Bundled model path (relative to this file)
_BUNDLED_MODEL_DIR = Path(__file__).parent / "models"


def _get_providers() -> List[str]:
    """Auto-select best available ONNX execution providers."""
    available = ort.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",
        "CoreMLExecutionProvider",
        "CPUExecutionProvider",
    ]
    return [p for p in preferred if p in available]


class PaddleDetector:
    """PaddleOCR DB text detector via ONNX Runtime.

    Usage:
        detector = PaddleDetector()
        results = detector([pil_image1, pil_image2])
        for result in results:
            for bbox in result.bboxes:
                print(bbox.polygon, bbox.confidence)
    """

    def __init__(self, cfg: PaddleDetectorConfig = PaddleDetectorConfig()):
        self.cfg = cfg
        model_path = cfg.model_path or self._resolve_model(cfg)

        providers = _get_providers()
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
        except Exception:
            log.warning(
                f"Failed to create ONNX session with {providers}, "
                "falling back to CPUExecutionProvider"
            )
            self.session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        active_provider = self.session.get_providers()[0]

        log.info(
            f"PaddleDetector ready | provider={active_provider} | "
            f"model={model_path}"
        )

    @staticmethod
    def _resolve_model(cfg: PaddleDetectorConfig) -> str:
        """Try bundled model first, then download from HuggingFace."""
        bundled = _BUNDLED_MODEL_DIR / cfg.bundled_filename
        if bundled.exists():
            log.info(f"Using bundled model: {bundled}")
            return str(bundled)

        from huggingface_hub import hf_hub_download
        log.info(
            f"Bundled model not found at {bundled}, "
            f"downloading from {cfg.model_repo}/{cfg.model_filename}"
        )
        return hf_hub_download(
            repo_id=cfg.model_repo, filename=cfg.model_filename
        )

    def _preprocess(
        self, img: Image.Image
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """PIL Image → normalized NCHW float32 array + original (w, h)."""
        rgb = np.array(img.convert("RGB"))
        orig_h, orig_w = rgb.shape[:2]

        # Resize preserving aspect ratio
        ratio = min(self.cfg.max_side_len / max(orig_h, orig_w), 1.0)
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)
        # Pad to multiple of 32
        new_h = math.ceil(new_h / 32) * 32
        new_w = math.ceil(new_w / 32) * 32

        resized = cv2.resize(rgb, (new_w, new_h))

        # Normalize with ImageNet mean/std
        mean = np.array(self.cfg.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.cfg.std, dtype=np.float32).reshape(1, 1, 3)
        normalized = (resized.astype(np.float32) / 255.0 - mean) / std

        # HWC → NCHW
        blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, (orig_w, orig_h)

    def __call__(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
    ) -> List[TextDetectionResult]:
        """Detect text regions in a list of PIL images.

        Args:
            images: List of PIL Images.
            batch_size: Unused (kept for API compatibility). Images are
                        processed individually due to variable sizes.

        Returns:
            List of TextDetectionResult, one per input image.
        """
        results = []
        t_total = time.time()

        for i, img in enumerate(images):
            t0 = time.time()

            blob, orig_size = self._preprocess(img)
            t1 = time.time()

            outputs = self.session.run(
                [self.output_name], {self.input_name: blob}
            )
            t2 = time.time()

            # Output shape: (1, 1, H, W) → squeeze to (H, W)
            prob_map = outputs[0].squeeze()
            if prob_map.ndim == 3:
                prob_map = prob_map[0]  # handle (1, H, W) case

            result = db_postprocess(
                prob_map,
                orig_size,
                binary_thresh=self.cfg.binary_thresh,
                box_thresh=self.cfg.box_thresh,
                unclip_ratio=self.cfg.unclip_ratio,
                min_box_size=self.cfg.min_box_size,
                max_candidates=self.cfg.max_candidates,
            )
            t3 = time.time()

            log.debug(
                f"[paddle_det] img {i}: preprocess={t1-t0:.4f}s "
                f"inference={t2-t1:.4f}s postprocess={t3-t2:.4f}s | "
                f"{len(result.bboxes)} boxes"
            )
            results.append(result)

        log.debug(
            f"[paddle_det] total: {time.time()-t_total:.4f}s | "
            f"{len(images)} images"
        )
        return results
