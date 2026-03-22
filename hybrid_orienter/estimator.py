import logging
import time
import torch
from typing import List

from .config import HoughConfig, PeakConfig, HybridOrienterConfig
from .hough import (
    build_hough_angles, hough_accumulator,
    find_hough_peaks, HoughResult, PeakResult,
)

log = logging.getLogger(__name__)


class HybridEstimator:
    """
    Hough-based skew estimator that uses an external prior angle (from Surya)
    to filter peaks and select the dominant text-line orientation.
    """

    def __init__(self, cfg: HybridOrienterConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device

        self.theta = build_hough_angles(cfg.num_angles, device)

        self._hough_cfg = HoughConfig(
            num_angles  = cfg.num_angles,
            max_samples = cfg.max_samples,
        )
        self._peak_cfg = PeakConfig(
            num_peaks      = cfg.num_peaks,
            nms_rho_size   = cfg.nms_rho_size,
            nms_theta_size = cfg.nms_theta_size,
            min_vote_ratio = cfg.min_vote_ratio,
        )

    def _filter_by_prior(self, peaks: PeakResult, prior_deg: float) -> float:
        """
        Filter Hough peaks to those within ±margin_tolerance of the prior angle.
        Returns weighted median of survivors, or prior itself if none survive.
        """
        if peaks.num_peaks == 0:
            return prior_deg

        margin      = self.cfg.margin_tolerance
        skew_angles = 90.0 - peaks.theta_deg  # already on CPU from find_hough_peaks
        votes       = peaks.votes

        mask            = (skew_angles >= prior_deg - margin) & \
                          (skew_angles <= prior_deg + margin)
        filtered_angles = skew_angles[mask]
        filtered_votes  = votes[mask]

        if len(filtered_angles) == 0:
            return prior_deg

        sorted_angles, si = torch.sort(filtered_angles)
        sorted_weights    = filtered_votes[si] / filtered_votes[si].sum()
        cum_w             = torch.cumsum(sorted_weights, dim=0)
        idx               = torch.searchsorted(cum_w, torch.tensor(0.5)).clamp(
                                0, len(sorted_angles) - 1
                            )
        return sorted_angles[idx].item()

    def estimate(self, edge_map: torch.Tensor, prior_angle: float, index: int = -1) -> dict:
        """
        Estimate skew for a single edge map using an external prior angle.

        Returns dict with angle_deg, should_rotate, prior_angle.
        """
        t0 = time.time()
        hr    = hough_accumulator(edge_map, self.theta, self._hough_cfg)
        t1 = time.time()
        peaks = find_hough_peaks(hr, self._peak_cfg)
        t2 = time.time()
        angle = self._filter_by_prior(peaks, prior_angle)
        t3 = time.time()

        edge_count = (edge_map.squeeze() > 0.5).sum().item()
        log.debug(
            f"[estimator] img {index}: edges={edge_count} | "
            f"hough={t1-t0:.4f}s peaks={t2-t1:.4f}s filter={t3-t2:.4f}s | "
            f"prior={prior_angle:.3f}° → angle={angle:.3f}°"
        )
        return {
            "angle_deg"    : angle,
            "should_rotate": abs(angle) > self.cfg.angle_tolerance,
            "prior_angle"  : prior_angle,
        }

    def estimate_batch(
        self,
        edge_maps    : List[torch.Tensor],
        prior_angles : List[float],
    ) -> List[dict]:
        t0 = time.time()
        results = [
            self.estimate(em, prior, index=i)
            for i, (em, prior) in enumerate(zip(edge_maps, prior_angles))
        ]
        log.debug(f"[estimator] estimate_batch total: {time.time()-t0:.4f}s | {len(edge_maps)} images")
        return results
