import numpy as np
import torch
import torch.nn.functional as F
from typing import List, NamedTuple

from .config import HoughConfig, PeakConfig


class HoughResult(NamedTuple):
    accum    : torch.Tensor   # (num_rho, num_angles) float32
    theta    : torch.Tensor   # (num_angles,) radians
    diag_len : int


class PeakResult(NamedTuple):
    theta_rad : torch.Tensor  # (K,) radians
    theta_deg : torch.Tensor  # (K,) degrees
    votes     : torch.Tensor  # (K,) vote counts
    num_peaks : int


def build_hough_angles(num_angles: int, device: torch.device) -> torch.Tensor:
    """Linearly spaced angles from 0.1° to 179.9° in radians."""
    return torch.linspace(
        0.1  * torch.pi / 180,
        179.9 * torch.pi / 180,
        num_angles,
        device = device,
        dtype  = torch.float32,
    )


def hough_accumulator(
    edge_map : torch.Tensor,
    theta    : torch.Tensor,
    cfg      : HoughConfig,
) -> HoughResult:
    """
    Build Hough line accumulator on-device using scatter_add_.
    ρ = x·cos(θ) + y·sin(θ)
    """
    device = edge_map.device
    A      = theta.shape[0]
    em     = edge_map.squeeze()
    H, W   = em.shape

    diag_len = int(torch.ceil(
        torch.sqrt(torch.tensor(H**2 + W**2, dtype=torch.float32))
    ).item())
    num_rho = 2 * diag_len + 1

    ys, xs = torch.where(em > 0.5)
    N = len(xs)

    if N == 0:
        return HoughResult(
            accum    = torch.zeros(num_rho, A, device=device),
            theta    = theta,
            diag_len = diag_len,
        )

    if N > cfg.max_samples:
        perm = torch.randperm(N, device=device)[:cfg.max_samples]
        xs, ys = xs[perm], ys[perm]
        N = cfg.max_samples

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    rho     = xs.float().unsqueeze(1) * cos_t + ys.float().unsqueeze(1) * sin_t
    rho_idx = (rho + diag_len).long().clamp(0, num_rho - 1)

    ang_idx  = torch.arange(A, device=device).unsqueeze(0).expand(N, -1)
    flat_idx = (rho_idx * A + ang_idx).reshape(-1)

    accum = torch.zeros(num_rho * A, device=device, dtype=torch.float32)
    accum.scatter_add_(0, flat_idx, torch.ones(N * A, device=device))
    accum = accum.view(num_rho, A)

    return HoughResult(accum=accum, theta=theta, diag_len=diag_len)


def hough_batch(
    edge_maps : List[torch.Tensor],
    device    : torch.device,
    cfg       : HoughConfig = HoughConfig(),
) -> List[HoughResult]:
    """Run hough_accumulator for each image in the batch."""
    theta = build_hough_angles(cfg.num_angles, device)
    return [hough_accumulator(em, theta, cfg) for em in edge_maps]


def find_hough_peaks(hr: HoughResult, cfg: PeakConfig) -> PeakResult:
    """
    Extract dominant angle peaks via max-pool NMS + topk.
    """
    accum  = hr.accum
    device = accum.device

    pooled = F.max_pool2d(
        accum.unsqueeze(0).unsqueeze(0),
        kernel_size = (cfg.nms_rho_size, cfg.nms_theta_size),
        stride      = 1,
        padding     = (cfg.nms_rho_size // 2, cfg.nms_theta_size // 2),
    ).squeeze()

    vote_threshold = cfg.min_vote_ratio * accum.max()
    peak_mask      = (accum == pooled) & (accum >= vote_threshold)

    rho_idx, ang_idx = torch.where(peak_mask)
    peak_votes       = accum[rho_idx, ang_idx]

    if len(peak_votes) == 0:
        empty = torch.tensor([], device=device)
        return PeakResult(empty, empty, empty, 0)

    k    = min(cfg.num_peaks, len(peak_votes))
    topk = torch.topk(peak_votes, k=k)

    best_ang  = ang_idx[topk.indices]
    best_t    = hr.theta[best_ang]

    return PeakResult(
        theta_rad = best_t,
        theta_deg = torch.rad2deg(best_t),
        votes     = topk.values,
        num_peaks = k,
    )