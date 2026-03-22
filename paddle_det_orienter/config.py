from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class HoughConfig:
    num_angles: int = 1024
    max_samples: int = 60_000


@dataclass
class PeakConfig:
    num_peaks: int = 50
    nms_rho_size: int = 9
    nms_theta_size: int = 9
    min_vote_ratio: float = 0.05


@dataclass
class TextAnglePriorConfig:
    min_confidence: float = 0.5
    max_skew_deg: float = 45.0


@dataclass
class PaddleDetectorConfig:
    model_path: Optional[str] = None
    # Bundled model (checked first, in paddle_det_orienter/models/)
    bundled_filename: str = "det_v3.onnx"
    # HuggingFace fallback (downloaded if bundled not found)
    model_repo: str = "monkt/paddleocr-onnx"
    model_filename: str = "detection/v3/det.onnx"
    # Preprocessing
    max_side_len: int = 960
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    # DB post-processing
    binary_thresh: float = 0.3
    box_thresh: float = 0.6
    unclip_ratio: float = 1.5
    min_box_size: int = 3
    max_candidates: int = 1000


@dataclass
class PaddleDetOrienterConfig:
    # Text angle prior
    paddle_min_confidence: float = 0.5
    paddle_max_skew_deg: float = 45.0
    # Preprocessing (Canny pipeline)
    blur_kernel: int = 11
    blur_sigma: float = 1.5
    canny_low: float = 0.05
    canny_high: float = 0.15
    # Hough
    num_angles: int = 1024
    max_samples: int = 60_000
    # Peak finding
    num_peaks: int = 50
    nms_rho_size: int = 9
    nms_theta_size: int = 9
    min_vote_ratio: float = 0.05
    # Angle logic
    margin_tolerance: float = 9.0
    angle_tolerance: float = 0.25
    # Interpolation — None = auto-select per device
    interp_mode: Optional[str] = None
    # Detector batch size — None = use device default
    detector_batch_size: Optional[int] = None
