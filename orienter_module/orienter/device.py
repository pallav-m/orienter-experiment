import torch


def get_device() -> torch.device:
    """
    Returns the best available device: CUDA > MPS > CPU.
    MPS = Apple Silicon GPU (M-series chips).
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def get_interp_mode(device: torch.device) -> str:
    """
    MPS does not support bicubic interpolation in grid_sample.
    Returns the best available interpolation mode for the given device.
    """
    if device.type == "mps":
        return "bilinear"
    return "bicubic"