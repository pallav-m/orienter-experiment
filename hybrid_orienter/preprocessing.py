import logging
import time
import cv2
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Union
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class ImageMeta:
    orig_h : int
    orig_w : int
    index  : int


def bgr_to_tensor(img_bgr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, 3) BGR uint8 → (1, 3, H, W) RGB float32 on device."""
    img_rgb = img_bgr[:, :, ::-1]  # BGR→RGB via slice (no cv2 overhead)
    t = torch.from_numpy(img_rgb.copy()).permute(2, 0, 1).float().div(255.0)
    return t.unsqueeze(0).to(device)


def pil_to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    """PIL Image → (1, 3, H, W) RGB float32 on device. No cv2."""
    rgb = np.asarray(img.convert("RGB"))
    t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0)
    return t.unsqueeze(0).to(device)


def to_grayscale(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) float32 → (B, 1, H, W) float32 grayscale."""
    weights = torch.tensor(
        [0.299, 0.587, 0.114],
        device=rgb_tensor.device,
    ).view(1, 3, 1, 1)
    return (rgb_tensor * weights).sum(dim=1, keepdim=True)


def build_batch(
    images : List[Union[np.ndarray, Image.Image]],
    device : torch.device,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[ImageMeta]]:
    """Convert list of BGR uint8 / PIL images to rgb + gray tensor lists."""
    rgb_tensors, gray_tensors, metas = [], [], []

    for i, img in enumerate(images):
        t0 = time.time()
        if isinstance(img, Image.Image):
            rgb = pil_to_tensor(img, device)
            h, w = img.height, img.width
            src = "PIL"
        else:
            assert img.dtype == np.uint8 and img.ndim == 3, \
                f"Image {i}: expected (H, W, 3) uint8, got {img.shape} {img.dtype}"
            rgb = bgr_to_tensor(img, device)
            h, w = img.shape[0], img.shape[1]
            src = "BGR"

        gray = to_grayscale(rgb)
        rgb_tensors.append(rgb)
        gray_tensors.append(gray)
        metas.append(ImageMeta(orig_h=h, orig_w=w, index=i))
        log.debug(f"[build_batch] img {i}: {src} {h}x{w} → tensor in {time.time()-t0:.4f}s")

    return rgb_tensors, gray_tensors, metas


def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) or (3, H, W) float32 [0,1] → (H, W, 3) BGR uint8."""
    t = t.squeeze(0).clamp(0, 1)
    rgb = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return rgb[:, :, ::-1].copy()  # RGB→BGR via slice (no cv2)


def blur_and_detect_edges(
    gray_tensors   : List[torch.Tensor],
    blur_kernel    : int   = 11,
    blur_sigma     : float = 1.5,
    low_threshold  : float = 0.05,
    high_threshold : float = 0.15,
) -> List[torch.Tensor]:
    """Blur + Canny via cv2 on CPU, then move edge maps back to device.

    cv2.Canny is ~27x faster than kornia.filters.canny on MPS and avoids
    the iterative hysteresis convergence loop that bottlenecks GPU batching.
    """
    assert blur_kernel % 2 == 1, "blur_kernel must be odd"

    # Convert normalized thresholds (0-1) to cv2 absolute thresholds (0-255)
    cv_low  = int(low_threshold * 255)
    cv_high = int(high_threshold * 255)

    results = []
    t_total = time.time()

    for i, g in enumerate(gray_tensors):
        t0 = time.time()
        device = g.device

        # (1, 1, H, W) float32 [0,1] → (H, W) uint8
        gray_np = (g.squeeze().cpu().numpy() * 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(gray_np, (blur_kernel, blur_kernel), blur_sigma)
        edges   = cv2.Canny(blurred, cv_low, cv_high)

        # (H, W) uint8 → (1, 1, H, W) float32 on original device
        edge_t = torch.from_numpy(edges).float().div(255.0).unsqueeze(0).unsqueeze(0).to(device)
        results.append(edge_t)

        log.debug(f"[blur_and_detect_edges] img {i}: {gray_np.shape[0]}x{gray_np.shape[1]} in {time.time()-t0:.4f}s")

    log.debug(f"[blur_and_detect_edges] total: {time.time()-t_total:.4f}s | {len(gray_tensors)} images")
    return results
