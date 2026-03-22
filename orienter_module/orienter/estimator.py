import torch
import torch.nn.functional as F
from typing import List

from .config import SkewEstimatorConfig, HoughConfig, PeakConfig
from .hough import (
    build_hough_angles, hough_accumulator,
    find_hough_peaks, HoughResult, PeakResult,
)


class SkewEstimator:
    """
    Pure-Hough skew estimator with rho-spread (multi-line density) prior.
    Robust to border lines and large non-text image elements.
    """

    def __init__(self, cfg: SkewEstimatorConfig, device: torch.device):
        self.cfg    = cfg
        self.device = device

        self.theta = build_hough_angles(cfg.num_angles, device)
        self.cos_t = torch.cos(self.theta)
        self.sin_t = torch.sin(self.theta)

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

    def _rho_peak_counts(self, hr: HoughResult) -> torch.Tensor:
        """Count distinct rho peaks per angle column — (A,) int tensor."""
        accum       = hr.accum
        vote_thresh = accum.max() * self.cfg.min_vote_ratio
        a_T         = accum.T.unsqueeze(1).float()

        nms = F.max_pool1d(
            a_T,
            kernel_size = self.cfg.nms_rho_size,
            stride      = 1,
            padding     = self.cfg.nms_rho_size // 2,
        )
        peak_mask = (a_T.squeeze(1) == nms.squeeze(1)) & \
                    (a_T.squeeze(1) >= vote_thresh)
        return peak_mask.sum(dim=1)

    def _find_dominant_cluster(
        self,
        peaks : PeakResult,
        hr    : HoughResult,
    ) -> float:
        """
        Use rho-spread as prior to select the text-line angle cluster.
        Falls back to prior directly if no peaks survive the margin filter.
        """
        if peaks.num_peaks == 0:
            return 0.0

        rho_counts     = self._rho_peak_counts(hr)
        best_ang_idx   = rho_counts.argmax().item()
        prior_skew_deg = 90.0 - torch.rad2deg(self.theta[best_ang_idx]).item()

        margin      = self.cfg.margin_tolerance
        skew_angles = (90.0 - peaks.theta_deg).cpu()
        votes       = peaks.votes.cpu()

        mask            = (skew_angles >= prior_skew_deg - margin) & \
                          (skew_angles <= prior_skew_deg + margin)
        filtered_angles = skew_angles[mask]
        filtered_votes  = votes[mask]

        if len(filtered_angles) == 0:
            return prior_skew_deg

        sorted_angles, si = torch.sort(filtered_angles)
        sorted_weights    = filtered_votes[si] / filtered_votes[si].sum()
        cum_w             = torch.cumsum(sorted_weights, dim=0)
        idx               = torch.searchsorted(cum_w, torch.tensor(0.5)).clamp(
                                0, len(sorted_angles) - 1
                            )
        return sorted_angles[idx].item()

    def estimate(self, edge_map: torch.Tensor) -> dict:
        hr    = hough_accumulator(edge_map, self.theta, self._hough_cfg)
        peaks = find_hough_peaks(hr, self._peak_cfg)
        angle = self._find_dominant_cluster(peaks, hr)
        return {
            "angle_deg"    : angle,
            "should_rotate": abs(angle) > self.cfg.angle_tolerance,
            "peaks"        : peaks,
            "hough_result" : hr,
        }

    def estimate_batch(self, edge_maps: List[torch.Tensor]) -> List[dict]:
        return [self.estimate(em) for em in edge_maps]