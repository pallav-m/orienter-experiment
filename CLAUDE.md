# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Orienter is a document skew detection and correction library with two implementations:

1. **Original (orienter.py)**: Uses OpenCV's EAST text detection model + scikit-image Hough transforms. Requires `frozen_east_text_detection.pb` model weights.
2. **PyTorch-native (orienter_module/)**: Pure PyTorch + Kornia pipeline, no EAST model needed. Supports CUDA, MPS (Apple Silicon), and CPU with auto-detection.

## Setup

```bash
# Install dependencies
pipenv install

# Install the orienter_module package in editable mode
cd orienter_module && pip install -e .
```

Requires Python >= 3.10.

## Architecture

### PyTorch Pipeline (orienter_module/orienter/)

Processing flow: `BGR uint8 → RGB tensor → Grayscale → Gaussian blur → Canny edges → Hough accumulator → rho-spread prior → angle clustering → bound-preserving rotation → BGR numpy`

Key modules:
- **orienter.py** — `TorchOrienter`: main class with `reorient()` and `batch_reorient()` entry points
- **estimator.py** — `SkewEstimator`: Hough-based angle estimation with rho-spread (multi-line density) prior and NMS peak selection
- **hough.py** — Pure-tensor Hough accumulator with memory guard (subsamples if >60k edge pixels)
- **config.py** — Nested dataclasses (`TorchOrienterConfig` > `SkewEstimatorConfig` > `HoughConfig`/`PeakConfig`) for all tunable parameters
- **preprocessing.py** — Image preprocessing (grayscale, blur, Canny via Kornia)
- **rotation.py** — Bound-preserving affine rotation
- **device.py** — Auto device selection (CUDA > MPS > CPU); MPS uses bilinear instead of bicubic interpolation

### Original Pipeline (orienter.py)

`Orienter` class combines EAST text detection angles with Hough line angles, filtering Hough results by EAST margin tolerance before rotation.

## Usage

```python
# PyTorch implementation
from orienter_module.orienter import TorchOrienter
orienter = TorchOrienter()  # auto-detects device
corrected = orienter.reorient(image)  # single image (numpy BGR)
results, angles = orienter.batch_reorient(images, return_angles=True)

# Original implementation
from orienter import Orienter
orienter = Orienter(east_model_path="frozen_east_text_detection.pb")
corrected, angle = orienter.re_orient_east(image)
```

## Notes

- No test suite, linter config, or CI pipeline exists currently.
- Both implementations share the same batch API: `batch_reorient(images, return_angles=False)` accepts file paths or numpy arrays.
