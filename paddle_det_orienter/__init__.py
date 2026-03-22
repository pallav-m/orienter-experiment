from .orienter import PaddleDetOrienter
from .config import PaddleDetOrienterConfig, PaddleDetectorConfig
from .detector import PaddleDetector
from .text_angle_prior import TextAnglePrior

__all__ = [
    "PaddleDetOrienter",
    "PaddleDetOrienterConfig",
    "PaddleDetectorConfig",
    "PaddleDetector",
    "TextAnglePrior",
]
