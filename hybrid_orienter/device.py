import torch


def get_device() -> torch.device:
    """Returns the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def get_interp_mode(device: torch.device) -> str:
    """MPS does not support bicubic in grid_sample."""
    if device.type == "mps":
        return "bilinear"
    return "bicubic"
