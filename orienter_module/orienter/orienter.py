import logging
import time
import cv2
import numpy as np
import torch
from typing import List, Optional, Tuple, Union

from .config import TorchOrienterConfig, SkewEstimatorConfig
from .device import get_device, get_interp_mode
from .preprocessing import (
    build_batch, tensor_to_bgr, to_grayscale,
    preprocess_for_edges, detect_edges_batch,
)
from .estimator import SkewEstimator
from .rotation import correct_skew_batch

log = logging.getLogger(__name__)


class TorchOrienter:
    """
    PyTorch-native document skew detection and correction.

    Pipeline:
      BGR numpy → tensor → grayscale → blur → Canny edges
      → Hough accumulator → rho-spread prior → angle cluster
      → bound-preserving rotation → BGR numpy

    No EAST model. No surya. No skimage. Pure PyTorch + kornia.
    """

    def __init__(
        self,
        cfg    : TorchOrienterConfig = TorchOrienterConfig(),
        device : Optional[torch.device] = None,
    ):
        self.cfg    = cfg
        self.device = device or get_device()
        self.interp_mode = cfg.interp_mode or get_interp_mode(self.device)

        if self.device.type == "mps":
            torch.set_default_dtype(torch.float32)

        self.estimator = SkewEstimator(
            SkewEstimatorConfig(
                num_angles       = cfg.num_angles,
                max_samples      = cfg.max_samples,
                num_peaks        = cfg.num_peaks,
                nms_rho_size     = cfg.nms_rho_size,
                nms_theta_size   = cfg.nms_theta_size,
                min_vote_ratio   = cfg.min_vote_ratio,
                margin_tolerance = cfg.margin_tolerance,
                angle_tolerance  = cfg.angle_tolerance,
            ),
            self.device,
        )

        log.info(
            f"TorchOrienter ready | device={self.device} | "
            f"interp={self.interp_mode}"
        )

    def _run_pipeline(
        self,
        images: List[np.ndarray],
    ) -> Tuple[List[torch.Tensor], List[float]]:
        """Core pipeline — shared by reorient() and batch_reorient()."""
        rgb_tensors, gray_tensors, _ = build_batch(images, self.device)

        blurred   = preprocess_for_edges(
            gray_tensors,
            blur_kernel = self.cfg.blur_kernel,
            blur_sigma  = self.cfg.blur_sigma,
        )
        edge_maps = detect_edges_batch(
            blurred,
            low_threshold  = self.cfg.canny_low,
            high_threshold = self.cfg.canny_high,
        )
        estimates  = self.estimator.estimate_batch(edge_maps)
        angle_degs = [e["angle_deg"] for e in estimates]

        corrected, applied = correct_skew_batch(
            rgb_tensors,
            angle_degs,
            self.device,
            angle_tolerance = self.cfg.angle_tolerance,
            interp_mode     = self.interp_mode,
        )
        return corrected, applied

    def reorient(
        self,
        image        : np.ndarray,
        return_angle : bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        Correct skew in a single BGR image.

        Args:
            image        : (H, W, 3) BGR uint8
            return_angle : also return the detected skew angle

        Returns:
            corrected BGR image, optionally with angle float
        """
        corrected, applied = self._run_pipeline([image])
        out = tensor_to_bgr(corrected[0])
        return (out, applied[0]) if return_angle else out

    def batch_reorient(
        self,
        images        : List[Union[np.ndarray, str]],
        return_angles : bool = False,
        verbose       : bool = True,
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[float]]]:
        """
        Correct skew in a batch of BGR images or file paths.
        API-compatible with the original Orienter.batch_reorient().

        Args:
            images        : list of BGR uint8 arrays or file path strings
            return_angles : also return list of detected angles
            verbose       : log progress every 10 images

        Returns:
            list of corrected BGR images, optionally with angles
        """
        results_out = [None] * len(images)
        angles_out  = [None] * len(images)
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
            return (results_out, angles_out) if return_angles else results_out

        t0 = time.time()
        corrected, applied = self._run_pipeline(loaded)

        for local_i, global_i in enumerate(valid_idx):
            results_out[global_i] = tensor_to_bgr(corrected[local_i])
            angles_out[global_i]  = applied[local_i]

            if verbose and (local_i + 1) % 10 == 0:
                elapsed = time.time() - t0
                avg     = elapsed / (local_i + 1)
                log.info(
                    f"Processed {local_i+1}/{len(loaded)} | "
                    f"avg {avg:.3f}s/img | "
                    f"ETA {avg * (len(loaded) - local_i - 1):.1f}s"
                )

        if verbose:
            total = time.time() - t0
            log.info(
                f"Batch done: {len(loaded)} imgs | "
                f"{total:.2f}s total | {total/len(loaded):.3f}s/img"
            )

        return (results_out, angles_out) if return_angles else results_out