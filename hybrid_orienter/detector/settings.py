import os
import sys
import torch
from pathlib import Path


def _default_cache_dir() -> str:
    """Match platformdirs user_cache_dir('datalab') without the dependency."""
    env = os.environ.get("MODEL_CACHE_DIR")
    if env:
        return env
    if sys.platform == "darwin":
        return str(Path.home() / "Library" / "Caches" / "datalab" / "models")
    elif sys.platform == "win32":
        return str(Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "datalab" / "Cache" / "models")
    else:
        return str(Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "datalab" / "models")


class Settings:
    """Simplified detection-only settings. No pydantic dependency."""

    def __init__(self):
        self.TORCH_DEVICE = os.environ.get("TORCH_DEVICE", None)
        self.DISABLE_TQDM = os.environ.get("DISABLE_TQDM", "").lower() in ("1", "true")
        self.S3_BASE_URL = os.environ.get("S3_BASE_URL", "https://models.datalab.to")
        self.PARALLEL_DOWNLOAD_WORKERS = int(os.environ.get("PARALLEL_DOWNLOAD_WORKERS", "10"))
        self.MODEL_CACHE_DIR = _default_cache_dir()

        # Detection-specific
        self.DETECTOR_BATCH_SIZE = None
        batch_env = os.environ.get("DETECTOR_BATCH_SIZE")
        if batch_env is not None:
            self.DETECTOR_BATCH_SIZE = int(batch_env)

        self.DETECTOR_MODEL_CHECKPOINT = os.environ.get(
            "DETECTOR_MODEL_CHECKPOINT", "s3://text_detection/2025_05_07"
        )
        self.DETECTOR_IMAGE_CHUNK_HEIGHT = int(
            os.environ.get("DETECTOR_IMAGE_CHUNK_HEIGHT", "1400")
        )
        self.DETECTOR_TEXT_THRESHOLD = float(
            os.environ.get("DETECTOR_TEXT_THRESHOLD", "0.6")
        )
        self.DETECTOR_BLANK_THRESHOLD = float(
            os.environ.get("DETECTOR_BLANK_THRESHOLD", "0.35")
        )
        self.DETECTOR_BOX_Y_EXPAND_MARGIN = float(
            os.environ.get("DETECTOR_BOX_Y_EXPAND_MARGIN", "0.05")
        )

    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        if self.TORCH_DEVICE_MODEL == "cpu":
            return torch.float32
        return torch.float16

    @property
    def INFERENCE_MODE(self):
        return torch.inference_mode


settings = Settings()
