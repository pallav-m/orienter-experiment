import logging
import sys

# Configure package logger to write to stdout with timestamps
_logger = logging.getLogger("hybrid_orienter")
if not _logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG)

from .orienter import HybridOrienter
from .config import HybridOrienterConfig, SuryaPriorConfig
from .surya_prior import SuryaPrior
from .device import get_device

__all__ = [
    "HybridOrienter",
    "HybridOrienterConfig",
    "SuryaPriorConfig",
    "SuryaPrior",
    "get_device",
]
