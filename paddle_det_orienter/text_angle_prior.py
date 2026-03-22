import math
import logging
import time
from typing import List, Optional

from PIL import Image

from .config import TextAnglePriorConfig

log = logging.getLogger(__name__)


class TextAnglePrior:
    """
    Compute a text-line angle prior from detected text polygons.

    The detector returns rotated quadrilateral polygons for each detected
    text region. The angle of the top edge (p0→p1) gives the text line
    orientation. The median angle across all detections serves as the
    prior for Hough-based refinement.
    """

    def __init__(
        self,
        predictor=None,
        cfg: TextAnglePriorConfig = TextAnglePriorConfig(),
    ):
        if predictor is None:
            raise ValueError("A text detector (predictor) must be provided")
        self._predictor = predictor
        self.cfg = cfg

    @staticmethod
    def _polygon_angle_deg(polygon: list) -> float:
        """Angle of the top edge (p0→p1) in degrees."""
        p0, p1 = polygon[0], polygon[1]
        return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

    def _angles_from_result(self, result) -> List[float]:
        """Extract valid skew angles from one TextDetectionResult."""
        angles = []
        for bbox in result.bboxes:
            if bbox.confidence is not None and bbox.confidence < self.cfg.min_confidence:
                continue
            angle = self._polygon_angle_deg(bbox.polygon)
            if abs(angle) <= self.cfg.max_skew_deg:
                angles.append(angle)
        return angles

    @staticmethod
    def _median(values: List[float]) -> float:
        """Median without numpy dependency."""
        values.sort()
        n = len(values)
        mid = n // 2
        if n % 2 == 0:
            return (values[mid - 1] + values[mid]) / 2.0
        return values[mid]

    def compute(self, pil_image: Image.Image) -> float:
        """Single-image prior angle via text detection."""
        results = self._predictor([pil_image])
        angles = self._angles_from_result(results[0])
        if not angles:
            log.debug("TextAnglePrior: no valid text angles found, returning 0.0")
            return 0.0
        return self._median(angles)

    def compute_batch(self, pil_images: List[Image.Image], batch_size: Optional[int] = None) -> List[float]:
        """Batch prior angles — single detector inference call."""
        t0 = time.time()
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        results = self._predictor(pil_images, **kwargs)
        t1 = time.time()
        log.debug(f"[text_angle_prior] predictor.__call__: {t1-t0:.4f}s | {len(pil_images)} images")

        priors = []
        for i, result in enumerate(results):
            angles = self._angles_from_result(result)
            if not angles:
                log.debug(f"[text_angle_prior] image {i} — no valid text angles, using 0.0")
                priors.append(0.0)
            else:
                log.debug(f"[text_angle_prior] image {i} — {len(angles)} text angles, median={self._median(list(angles)):.3f}°")
                priors.append(self._median(angles))
        t2 = time.time()
        log.debug(f"[text_angle_prior] angle extraction: {t2-t1:.4f}s | {len(pil_images)} images")
        return priors
