import logging
import time
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

from .config import PaddleDetOrienterConfig, PaddleDetectorConfig, TextAnglePriorConfig
from .device import get_device, get_interp_mode
from .estimator import SkewEstimator
from .preprocessing import build_batch, tensor_to_bgr, blur_and_detect_edges
from .rotation import correct_skew_batch
from .text_angle_prior import TextAnglePrior
from .detector import PaddleDetector

log = logging.getLogger(__name__)


class PaddleDetOrienter:
    """
    GPU-accelerated document skew detection and correction using
    PaddleOCR DB text detection (ONNX) as prior + Hough refinement.

    Pipeline:
      input → PaddleOCR DB detection → polygon angles → prior
      input → tensor → grayscale → blur → Canny edges
      → Hough accumulator → filter by prior → weighted median
      → (optional) bound-preserving rotation → BGR numpy
    """

    def __init__(
        self,
        cfg: PaddleDetOrienterConfig = PaddleDetOrienterConfig(),
        device: Optional[torch.device] = None,
        detector: Optional[PaddleDetector] = None,
        detector_cfg: Optional[PaddleDetectorConfig] = None,
    ):
        self.cfg = cfg
        self.device = device or get_device()
        self.interp_mode = cfg.interp_mode or get_interp_mode(self.device)

        if self.device.type == "mps":
            torch.set_default_dtype(torch.float32)

        paddle_det = detector or PaddleDetector(
            detector_cfg or PaddleDetectorConfig()
        )

        self.prior = TextAnglePrior(
            predictor=paddle_det,
            cfg=TextAnglePriorConfig(
                min_confidence=cfg.paddle_min_confidence,
                max_skew_deg=cfg.paddle_max_skew_deg,
            ),
        )

        self.estimator = SkewEstimator(cfg, self.device)

        log.info(
            f"PaddleDetOrienter ready | pipeline_device={self.device} | "
            f"interp={self.interp_mode}"
        )

    @staticmethod
    def _normalize_inputs(
        images: List[Union[np.ndarray, Image.Image]],
    ) -> Tuple[List[Union[np.ndarray, Image.Image]], List[Image.Image]]:
        """Return (originals for build_batch, PIL list for detector)."""
        pil_list = []
        for img in images:
            if isinstance(img, Image.Image):
                pil_list.append(img)
            else:
                pil_list.append(Image.fromarray(img[:, :, ::-1]))
        return images, pil_list

    def _run_pipeline(
        self,
        images: List[Union[np.ndarray, Image.Image]],
        return_corrected: bool = True,
    ) -> Tuple[Optional[List[torch.Tensor]], List[float]]:
        """Core pipeline."""
        n = len(images)
        t0 = time.time()

        originals, pil_images = self._normalize_inputs(images)
        t1 = time.time()
        log.debug(f"[pipeline] normalize_inputs: {t1-t0:.4f}s | {n} images")

        # PaddleOCR detection → prior angles
        prior_angles = self.prior.compute_batch(
            pil_images, batch_size=self.cfg.detector_batch_size
        )
        t2 = time.time()
        log.debug(
            f"[pipeline] prior.compute_batch: {t2-t1:.4f}s | "
            f"{n} images | priors={prior_angles}"
        )

        # GPU pipeline: build tensors → blur + Canny → Hough → filter
        rgb_tensors, gray_tensors, _ = build_batch(originals, self.device)
        t3 = time.time()
        log.debug(f"[pipeline] build_batch: {t3-t2:.4f}s | {n} images")

        edge_maps = blur_and_detect_edges(
            gray_tensors,
            blur_kernel=self.cfg.blur_kernel,
            blur_sigma=self.cfg.blur_sigma,
            low_threshold=self.cfg.canny_low,
            high_threshold=self.cfg.canny_high,
        )
        del gray_tensors
        t4 = time.time()
        log.debug(f"[pipeline] blur_and_detect_edges: {t4-t3:.4f}s | {n} edge maps")

        estimates = self.estimator.estimate_batch(edge_maps, prior_angles)
        angle_degs = [e["angle_deg"] for e in estimates]
        del edge_maps
        t5 = time.time()
        log.debug(
            f"[pipeline] estimator.estimate_batch: {t5-t4:.4f}s | "
            f"{n} images | angles={angle_degs}"
        )

        if not return_corrected:
            del rgb_tensors
            log.debug(
                f"[pipeline] TOTAL: {t5-t0:.4f}s | {n} images | "
                f"{(t5-t0)/n:.4f}s/img (angles only)"
            )
            return None, angle_degs

        corrected, applied = correct_skew_batch(
            rgb_tensors,
            angle_degs,
            self.device,
            angle_tolerance=self.cfg.angle_tolerance,
            interp_mode=self.interp_mode,
        )
        t6 = time.time()
        log.debug(f"[pipeline] correct_skew_batch: {t6-t5:.4f}s | {n} images")
        log.debug(
            f"[pipeline] TOTAL: {t6-t0:.4f}s | {n} images | "
            f"{(t6-t0)/n:.4f}s/img (with correction)"
        )
        return corrected, applied

    def reorient(
        self,
        image: Union[np.ndarray, Image.Image],
        return_corrected: bool = True,
    ) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect and optionally correct skew in a single image.

        Args:
            image            : BGR uint8 numpy array or PIL Image
            return_corrected : if True, also return the corrected image

        Returns:
            (corrected_image_or_None, angle_deg)
        """
        corrected_tensors, applied = self._run_pipeline(
            [image], return_corrected=return_corrected,
        )
        if return_corrected:
            return tensor_to_bgr(corrected_tensors[0]), applied[0]
        return None, applied[0]

    def batch_reorient(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        return_corrected: bool = True,
        verbose: bool = True,
    ) -> Tuple[Optional[List[np.ndarray]], List[float]]:
        """
        Detect and optionally correct skew in a batch of images.

        Args:
            images           : list of BGR uint8 arrays, PIL Images, or file path strings
            return_corrected : if True, also return corrected images
            verbose          : log timing info

        Returns:
            (list_of_corrected_or_None, list_of_angles)
        """
        angles_out = [None] * len(images)
        loaded, valid_idx = [], []

        for i, item in enumerate(images):
            if isinstance(item, str):
                img = cv2.imread(item)
                if img is None:
                    log.warning(f"Failed to load image: {item}")
                    continue
            else:
                img = item
            loaded.append(img)
            valid_idx.append(i)

        if not loaded:
            results_out = [None] * len(images) if return_corrected else None
            return results_out, angles_out

        t0 = time.time()
        corrected_tensors, applied = self._run_pipeline(
            loaded, return_corrected=return_corrected,
        )

        # Scatter results back to original indices
        results_out = [None] * len(images) if return_corrected else None
        for local_i, global_i in enumerate(valid_idx):
            angles_out[global_i] = applied[local_i]
            if return_corrected:
                results_out[global_i] = tensor_to_bgr(corrected_tensors[local_i])

        if verbose:
            total = time.time() - t0
            log.info(
                f"Batch done: {len(loaded)} imgs | "
                f"{total:.2f}s total | {total/len(loaded):.3f}s/img"
            )

        return results_out, angles_out
