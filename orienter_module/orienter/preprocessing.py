import cv2
import numpy as np
import torch
import torch.nn.functional as F
import kornia.filters as KF
from dataclasses import dataclass
from typing import List, Tuple

from .config import TorchOrienterConfig
from .device import get_device


@dataclass
class ImageMeta:
    orig_h : int
    orig_w : int
    index  : int


def bgr_to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) BGR uint8 → (1, 3, H, W) RGB float32 on device."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0)
    return t.unsqueeze(0).to(device)


def to_grayscale(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) float32 → (B, 1, H, W) float32 grayscale."""
    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        device=rgb_tensor.device
    ).view(1, 3, 1, 1)
    return (rgb_tensor * weights).sum(dim=1, keepdim=True)


def build_batch(
    images : List[np.ndarray],
    device : torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[ImageMeta]]:
    """
    Convert list of BGR uint8 images to rgb + gray tensor lists.
    Images kept at original size (no forced padding).
    """
    rgb_tensors, gray_tensors, metas = [], [], []

    for i, img in enumerate(images):
        assert img.dtype == np.uint8 and img.ndim == 3, \
            f"Image {i}: expected (H, W, 3) uint8, got {img.shape} {img.dtype}"

        rgb  = bgr_to_tensor(img, device)
        gray = to_grayscale(rgb)

        rgb_tensors.append(rgb)
        gray_tensors.append(gray)
        metas.append(ImageMeta(orig_h=img.shape[0], orig_w=img.shape[1], index=i))

    return rgb_tensors, gray_tensors, metas


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) or (3, H, W) float32 [0,1] → (H, W, 3) BGR uint8."""
    t = t.squeeze(0).clamp(0, 1)
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preprocess_for_edges(
    gray_tensors : List[torch.Tensor],
    blur_kernel  : int   = 11,
    blur_sigma   : float = 1.5,
) -> List[torch.Tensor]:
    """Apply Gaussian blur to each (1, 1, H, W) gray tensor."""
    assert blur_kernel % 2 == 1, "blur_kernel must be odd"
    return [
        KF.gaussian_blur2d(
            g,
            kernel_size = (blur_kernel, blur_kernel),
            sigma       = (blur_sigma, blur_sigma),
        )
        for g in gray_tensors
    ]


def detect_edges_batch(
    blurred_tensors : List[torch.Tensor],
    low_threshold   : float = 0.05,
    high_threshold  : float = 0.15,
) -> List[torch.Tensor]:
    """
    Run kornia Canny on each (1, 1, H, W) blurred tensor.
    Returns list of (1, 1, H, W) binary float32 edge maps.
    """
    edge_maps = []
    for b in blurred_tensors:
        _, edges = KF.canny(
            b,
            low_threshold  = low_threshold,
            high_threshold = high_threshold,
            kernel_size    = (5, 5),
            sigma          = (1.0, 1.0),
            hysteresis     = True,
        )
        edge_maps.append(edges)
    return edge_maps