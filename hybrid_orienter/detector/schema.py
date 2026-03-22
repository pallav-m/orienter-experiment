from dataclasses import dataclass, field
from typing import List, Optional, Any

from .polygon import PolygonBox


@dataclass
class TextDetectionResult:
    bboxes: List[PolygonBox]
    heatmap: Optional[Any] = None
    affinity_map: Optional[Any] = None
    image_bbox: List[float] = field(default_factory=list)
