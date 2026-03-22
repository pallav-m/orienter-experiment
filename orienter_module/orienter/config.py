from dataclasses import dataclass
from typing import Optional


@dataclass
class HoughConfig:
    num_angles  : int = 1024    # angular resolution — more = finer angle detection
    max_samples : int = 60_000  # subsample edge pixels above this (memory guard)


@dataclass
class PeakConfig:
    num_peaks        : int   = 50    # max peaks to extract from accumulator
    nms_rho_size     : int   = 9     # NMS suppression window along rho axis
    nms_theta_size   : int   = 9     # NMS suppression window along theta axis
    min_vote_ratio   : float = 0.05  # peak must have >= this fraction of max votes


@dataclass
class SkewEstimatorConfig:
    # Hough
    num_angles       : int   = 1024
    max_samples      : int   = 60_000
    # Peak finding
    num_peaks        : int   = 50
    nms_rho_size     : int   = 9
    nms_theta_size   : int   = 9
    min_vote_ratio   : float = 0.05
    # Angle logic
    margin_tolerance : float = 9.0   # degrees — cluster filter window
    angle_tolerance  : float = 0.25  # degrees — skip rotation below this


@dataclass
class TorchOrienterConfig:
    # Preprocessing
    blur_kernel      : int   = 11
    blur_sigma       : float = 1.5
    canny_low        : float = 0.05
    canny_high       : float = 0.15
    # Hough
    num_angles       : int   = 1024
    max_samples      : int   = 60_000
    # Peak finding
    num_peaks        : int   = 50
    nms_rho_size     : int   = 9
    nms_theta_size   : int   = 9
    min_vote_ratio   : float = 0.05
    # Angle logic
    margin_tolerance : float = 9.0
    angle_tolerance  : float = 0.25
    # Interpolation — None = auto-select per device
    interp_mode      : Optional[str] = None