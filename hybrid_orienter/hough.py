import math
import torch
import torch.nn.functional as F
from typing import NamedTuple

from .config import HoughConfig, PeakConfig


class HoughResult(NamedTuple):
    accum    : torch.Tensor   # (num_rho, num_angles) float32
    theta    : torch.Tensor   # (num_angles,) radians
    diag_len : int


class PeakResult(NamedTuple):
    theta_rad : torch.Tensor  # (K,) radians  — on CPU
    theta_deg : torch.Tensor  # (K,) degrees  — on CPU
    votes     : torch.Tensor  # (K,) vote counts — on CPU
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
    """Build Hough line accumulator on-device using scatter_add_."""
    device = edge_map.device
    A      = theta.shape[0]
    em     = edge_map.squeeze()
    H, W   = em.shape

    diag_len = math.ceil(math.sqrt(H * H + W * W))
    num_rho  = 2 * diag_len + 1

    # torch.nonzero is slightly more efficient than torch.where
    coords = torch.nonzero(em > 0.5, as_tuple=False)  # (N, 2)
    N = coords.shape[0]

    if N == 0:
        return HoughResult(
            accum    = torch.zeros(num_rho, A, device=device),
            theta    = theta,
            diag_len = diag_len,
        )

    ys, xs = coords[:, 0], coords[:, 1]

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


def find_hough_peaks(hr: HoughResult, cfg: PeakConfig) -> PeakResult:
    """Extract dominant angle peaks via fused NMS + topk (single GPU sync)."""
    accum  = hr.accum
    device = accum.device
    A      = hr.theta.shape[0]

    pooled = F.max_pool2d(
        accum.unsqueeze(0).unsqueeze(0),
        kernel_size = (cfg.nms_rho_size, cfg.nms_theta_size),
        stride      = 1,
        padding     = (cfg.nms_rho_size // 2, cfg.nms_theta_size // 2),
    ).squeeze()

    # Zero out non-local-maxima, then topk on flattened tensor
    # This avoids separate .max() and torch.where syncs
    masked = accum * (accum == pooled).float()
    flat   = masked.view(-1)

    k = min(cfg.num_peaks, flat.numel())
    topk_vals, topk_idx = torch.topk(flat, k=k)

    # Single GPU→CPU transfer for all peak data
    topk_vals_cpu = topk_vals.cpu()
    topk_idx_cpu  = topk_idx.cpu()
    theta_cpu     = hr.theta.cpu()

    # Filter by vote threshold on CPU (no GPU sync needed)
    max_val = topk_vals_cpu[0].item()
    if max_val == 0:
        empty = torch.tensor([])
        return PeakResult(empty, empty, empty, 0)

    threshold = cfg.min_vote_ratio * max_val
    valid     = topk_vals_cpu >= threshold

    topk_vals_cpu = topk_vals_cpu[valid]
    topk_idx_cpu  = topk_idx_cpu[valid]

    if len(topk_vals_cpu) == 0:
        empty = torch.tensor([])
        return PeakResult(empty, empty, empty, 0)

    ang_idx = topk_idx_cpu % A
    best_t  = theta_cpu[ang_idx]

    return PeakResult(
        theta_rad = best_t,
        theta_deg = torch.rad2deg(best_t),
        votes     = topk_vals_cpu,
        num_peaks = len(topk_vals_cpu),
    )
