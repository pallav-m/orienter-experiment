import logging
from typing import List, Tuple

import cv2
import numpy as np
import pyclipper

from .schema import PolygonBox, TextDetectionResult

log = logging.getLogger(__name__)


def _unclip(polygon: np.ndarray, unclip_ratio: float) -> np.ndarray:
    """Expand a shrunk polygon using Vatti clipping (pyclipper).

    Uses cv2 for area/perimeter (fast C++) instead of shapely (slow Python).
    """
    area = abs(float(cv2.contourArea(polygon)))
    if area == 0:
        return polygon
    peri = float(cv2.arcLength(polygon, True))
    distance = area * unclip_ratio / (peri + 1e-6)
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(
        pyclipper.scale_to_clipper(polygon.tolist()),
        pyclipper.JT_MITER,
        pyclipper.ET_CLOSEDPOLYGON,
    )
    expanded = offset.Execute(pyclipper.scale_to_clipper(distance))
    if not expanded:
        return polygon
    return np.array(pyclipper.scale_from_clipper(expanded[0]), dtype=np.float32)


def _order_points(box: np.ndarray) -> np.ndarray:
    """Order 4-point polygon: top-left first, then clockwise.

    Ensures p0→p1 is the top edge, so TextAnglePrior._polygon_angle_deg
    computes the correct text-line angle.
    """
    startidx = box.sum(axis=1).argmin()
    return np.roll(box, 4 - startidx, 0)


def _box_score(prob_map: np.ndarray, polygon: np.ndarray) -> float:
    """Mean probability within the polygon region (fast mode)."""
    h, w = prob_map.shape
    x_min = np.clip(np.floor(polygon[:, 0].min()).astype(int), 0, w - 1)
    x_max = np.clip(np.ceil(polygon[:, 0].max()).astype(int), 0, w - 1)
    y_min = np.clip(np.floor(polygon[:, 1].min()).astype(int), 0, h - 1)
    y_max = np.clip(np.ceil(polygon[:, 1].max()).astype(int), 0, h - 1)

    mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
    shifted = polygon.copy()
    shifted[:, 0] -= x_min
    shifted[:, 1] -= y_min
    cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)

    roi = prob_map[y_min:y_max + 1, x_min:x_max + 1]
    if mask.sum() == 0:
        return 0.0
    return float(cv2.mean(roi, mask)[0])


def db_postprocess(
    prob_map: np.ndarray,
    orig_size: Tuple[int, int],
    binary_thresh: float = 0.3,
    box_thresh: float = 0.6,
    unclip_ratio: float = 1.5,
    min_box_size: int = 3,
    max_candidates: int = 1000,
) -> TextDetectionResult:
    """Convert DB model probability map to TextDetectionResult.

    Args:
        prob_map: (H, W) float32 probability map from ONNX model.
        orig_size: (width, height) of the original image.
        binary_thresh: Threshold for binarization.
        box_thresh: Minimum mean-probability within polygon to keep.
        unclip_ratio: Vatti clipping expansion ratio.
        min_box_size: Discard boxes with short side < this.
        max_candidates: Maximum contours to process.

    Returns:
        TextDetectionResult with detected PolygonBox instances.
    """
    pred_h, pred_w = prob_map.shape
    orig_w, orig_h = orig_size

    # Binarize
    bitmap = (prob_map > binary_thresh).astype(np.uint8)

    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours[:max_candidates]:
        if len(contour) < 4:
            continue

        # Score on the original probability map
        contour_squeezed = contour.squeeze(1).astype(np.float32)
        score = _box_score(prob_map, contour_squeezed)
        if score < box_thresh:
            continue

        # Unclip
        expanded = _unclip(contour_squeezed, unclip_ratio)
        if len(expanded) < 4:
            continue

        # Get clean rotated rectangle from expanded polygon
        rect = cv2.minAreaRect(expanded.astype(np.int32))
        box = cv2.boxPoints(rect)

        # Filter small boxes
        short_side = min(rect[1])
        if short_side < min_box_size:
            continue

        # Order points: top-left first, clockwise
        box = _order_points(box)

        # Rescale to original image coordinates
        box[:, 0] = np.clip(box[:, 0] * (orig_w / pred_w), 0, orig_w)
        box[:, 1] = np.clip(box[:, 1] * (orig_h / pred_h), 0, orig_h)

        boxes.append(PolygonBox(polygon=box, confidence=score))

    log.debug(f"[db_postprocess] {len(boxes)} boxes from {len(contours)} contours")
    return TextDetectionResult(
        bboxes=boxes,
        image_bbox=[0, 0, orig_w, orig_h],
    )
