import copy
import numbers
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class PolygonBox:
    """Simplified polygon box — no pydantic dependency."""

    polygon: List[List[float]]
    confidence: Optional[float] = None

    def __post_init__(self):
        self.polygon = self._normalize_polygon(self.polygon)

    @staticmethod
    def _normalize_polygon(value):
        if isinstance(value, np.ndarray):
            if value.shape == (4, 2):
                return value.tolist()
        if isinstance(value, (list, tuple)) and len(value) == 4:
            if all(isinstance(x, numbers.Number) for x in value):
                x_min, y_min, x_max, y_max = [float(v) for v in value]
                return [
                    [x_min, y_min], [x_max, y_min],
                    [x_max, y_max], [x_min, y_max],
                ]
            if all(isinstance(p, (list, tuple)) and len(p) == 2 for p in value):
                return [[float(v) for v in p] for p in value]
        raise ValueError(
            f"Input must be bbox [x_min,y_min,x_max,y_max] or 4-corner polygon. Got {value}"
        )

    @property
    def bbox(self) -> List[float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return [min(xs), min(ys), max(xs), max(ys)]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def area(self):
        return self.width * self.height

    def rescale(self, processor_size, image_size):
        page_width, page_height = processor_size
        img_width, img_height = image_size
        width_scaler = img_width / page_width
        height_scaler = img_height / page_height
        for corner in self.polygon:
            corner[0] = int(corner[0] * width_scaler)
            corner[1] = int(corner[1] * height_scaler)

    def fit_to_bounds(self, bounds):
        new_corners = copy.deepcopy(self.polygon)
        for corner in new_corners:
            corner[0] = max(min(corner[0], bounds[2]), bounds[0])
            corner[1] = max(min(corner[1], bounds[3]), bounds[1])
        self.polygon = new_corners

    def expand(self, x_margin: float, y_margin: float):
        new_polygon = []
        x_m = x_margin * self.width
        y_m = y_margin * self.height
        for idx, poly in enumerate(self.polygon):
            if idx == 0:
                new_polygon.append([int(poly[0] - x_m), int(poly[1] - y_m)])
            elif idx == 1:
                new_polygon.append([int(poly[0] + x_m), int(poly[1] - y_m)])
            elif idx == 2:
                new_polygon.append([int(poly[0] + x_m), int(poly[1] + y_m)])
            elif idx == 3:
                new_polygon.append([int(poly[0] - x_m), int(poly[1] + y_m)])
        self.polygon = new_polygon
