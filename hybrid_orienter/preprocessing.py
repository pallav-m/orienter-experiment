import cv2
import numpy as np
import torch
import kornia.filters as KF
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Union
from PIL import Image


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
        if isinstance(img, Image.Image):
            rgb = pil_to_tensor(img, device)
            h, w = img.height, img.width
        else:
            assert img.dtype == np.uint8 and img.ndim == 3, \
                f"Image {i}: expected (H, W, 3) uint8, got {img.shape} {img.dtype}"
            rgb = bgr_to_tensor(img, device)
            h, w = img.shape[0], img.shape[1]

        gray = to_grayscale(rgb)
        rgb_tensors.append(rgb)
        gray_tensors.append(gray)
        metas.append(ImageMeta(orig_h=h, orig_w=w, index=i))

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
    """Size-grouped batched blur + Canny. Same-size images processed in one GPU call."""
    assert blur_kernel % 2 == 1, "blur_kernel must be odd"

    # Group tensors by (H, W) for batched processing
    groups = defaultdict(list)  # (H, W) -> [(original_index, tensor)]
    for i, g in enumerate(gray_tensors):
        key = (g.shape[-2], g.shape[-1])
        groups[key].append((i, g))

    results = [None] * len(gray_tensors)

    for (_h, _w), items in groups.items():
        indices, tensors = zip(*items)

        # Stack into (N, 1, H, W) — single blur + Canny call per size group
        batch = torch.cat(tensors, dim=0)

        blurred = KF.gaussian_blur2d(
            batch,
            kernel_size=(blur_kernel, blur_kernel),
            sigma=(blur_sigma, blur_sigma),
        )
        _, edges = KF.canny(
            blurred,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            kernel_size=(5, 5),
            sigma=(1.0, 1.0),
            hysteresis=True,
        )

        # Scatter results back to original order
        for j, idx in enumerate(indices):
            results[idx] = edges[j:j+1]

    return results
