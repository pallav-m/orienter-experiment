import numpy as np
import torch
import torch.nn.functional as F
import kornia.geometry.transform as KGT
from typing import List, Tuple


def rotate_bound_tensor(
    img_tensor  : torch.Tensor,
    angle_deg   : float,
    device      : torch.device,
    interp_mode : str = "bilinear",
) -> torch.Tensor:
    """
    Bound-preserving rotation — MPS-safe.
    Negates angle to match cv2.getRotationMatrix2D(center, -angle, 1.0).
    Pre-pads with replicate border to avoid padding_mode='border' (unsupported on MPS).
    """
    _, _, H, W = img_tensor.shape
    angle_rad  = float(-angle_deg * np.pi / 180.0)
    cos_a      = np.cos(angle_rad)
    sin_a      = np.sin(angle_rad)

    nW = int(abs(H * sin_a) + abs(W * cos_a))
    nH = int(abs(H * cos_a) + abs(W * sin_a))

    pad    = int(np.ceil(
        np.sqrt((nW / 2) ** 2 + (nH / 2) ** 2) -
        np.sqrt((W  / 2) ** 2 + (H  / 2) ** 2)
    )) + 4
    padded = F.pad(img_tensor, [pad, pad, pad, pad], mode='replicate')
    _, _, pH, pW = padded.shape
    pcX, pcY = pW / 2.0, pH / 2.0

    tx = (nW / 2.0) - cos_a * pcX - sin_a * pcY
    ty = (nH / 2.0) + sin_a * pcX - cos_a * pcY

    M = torch.tensor(
        [[ cos_a,  sin_a, tx],
         [-sin_a,  cos_a, ty]],
        dtype=torch.float32, device=device
    ).unsqueeze(0)

    return KGT.warp_affine(
        padded, M,
        dsize         = (nH, nW),
        mode          = interp_mode,
        padding_mode  = 'zeros',
        align_corners = True,
    )


def correct_skew(
    img_tensor      : torch.Tensor,
    angle_deg       : float,
    device          : torch.device,
    angle_tolerance : float = 0.25,
    interp_mode     : str   = "bilinear",
) -> Tuple[torch.Tensor, float]:
    if abs(angle_deg) <= angle_tolerance:
        return img_tensor, 0.0
    return rotate_bound_tensor(img_tensor, angle_deg, device, interp_mode), angle_deg


def correct_skew_batch(
    rgb_tensors     : List[torch.Tensor],
    angle_degs      : List[float],
    device          : torch.device,
    angle_tolerance : float = 0.25,
    interp_mode     : str   = "bilinear",
) -> Tuple[List[torch.Tensor], List[float]]:
    corrected, applied = [], []
    for t, a in zip(rgb_tensors, angle_degs):
        c, aa = correct_skew(t, a, device, angle_tolerance, interp_mode)
        corrected.append(c)
        applied.append(aa)
    return corrected, applied
