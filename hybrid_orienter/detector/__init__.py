import logging
import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

from .settings import settings
from .schema import TextDetectionResult
from .polygon import PolygonBox
from .heatmap import parallel_get_boxes
from .processor import SegformerImageProcessor
from .model.config import EfficientViTConfig
from .model.encoderdecoder import EfficientViTForSemanticSegmentation

log = logging.getLogger(__name__)

__all__ = ["TextDetector", "TextDetectionResult", "PolygonBox"]

# Default batch sizes per device
_DEFAULT_BATCH_SIZES = {"cpu": 8, "mps": 8, "cuda": 36}


def _get_total_splits(image_size, height):
    img_height = list(image_size)[1]
    if img_height > settings.DETECTOR_IMAGE_CHUNK_HEIGHT:
        return math.ceil(img_height / height)
    return 1


def _split_image(img, height):
    img_height = list(img.size)[1]
    if img_height > settings.DETECTOR_IMAGE_CHUNK_HEIGHT:
        num_splits = math.ceil(img_height / height)
        splits = []
        split_heights = []
        for i in range(num_splits):
            top = i * height
            bottom = min((i + 1) * height, img_height)
            cropped = img.crop((0, top, img.size[0], bottom))
            chunk_height = bottom - top
            if chunk_height < height:
                cropped = ImageOps.pad(cropped, (img.size[0], height), color=255, centering=(0, 0))
            splits.append(cropped)
            split_heights.append(chunk_height)
        return splits, split_heights
    return [img.copy()], [img_height]


class TextDetector:
    """
    Standalone text detector — extracted from surya DetectionPredictor.

    Usage:
        detector = TextDetector()
        results = detector([pil_image1, pil_image2])
        for result in results:
            for bbox in result.bboxes:
                print(bbox.polygon, bbox.confidence)
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        checkpoint = checkpoint or settings.DETECTOR_MODEL_CHECKPOINT
        device = device or settings.TORCH_DEVICE_MODEL
        dtype = dtype or settings.MODEL_DTYPE

        config = EfficientViTConfig.from_pretrained(checkpoint)
        self.model = EfficientViTForSemanticSegmentation.from_pretrained(
            checkpoint, dtype=dtype, config=config,
        )
        self.model = self.model.to(device).eval()
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)

        self._device = device
        self._dtype = dtype
        self._disable_tqdm = settings.DISABLE_TQDM

        log.info(
            f"TextDetector ready | device={device} | dtype={dtype} | "
            f"batch_size={self._get_batch_size()} | "
            f"checkpoint={checkpoint}"
        )

    @property
    def disable_tqdm(self) -> bool:
        return self._disable_tqdm

    @disable_tqdm.setter
    def disable_tqdm(self, value: bool):
        self._disable_tqdm = bool(value)

    def _get_batch_size(self) -> int:
        if settings.DETECTOR_BATCH_SIZE is not None:
            return settings.DETECTOR_BATCH_SIZE
        return _DEFAULT_BATCH_SIZES.get(settings.TORCH_DEVICE_MODEL, 8)

    def _prepare_image(self, img: Image.Image) -> torch.Tensor:
        new_size = (self.processor.size["width"], self.processor.size["height"])
        img.thumbnail(new_size, Image.Resampling.LANCZOS)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        img = np.asarray(img, dtype=np.uint8)
        img = self.processor(img)["pixel_values"][0]
        return torch.from_numpy(img)

    def __call__(
        self,
        images: List[Image.Image],
        batch_size: Optional[int] = None,
        include_maps: bool = False,
    ) -> List[TextDetectionResult]:
        assert all(isinstance(img, Image.Image) for img in images)
        batch_size = batch_size or self._get_batch_size()
        heatmap_count = self.model.config.num_labels

        orig_sizes = [img.size for img in images]
        splits_per_image = [
            _get_total_splits(size, self.processor.size["height"])
            for size in orig_sizes
        ]

        # Build batches respecting split counts
        batches = []
        current_batch = []
        current_size = 0
        for i in range(len(images)):
            if current_size + splits_per_image[i] > batch_size and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
            current_batch.append(i)
            current_size += splits_per_image[i]
        if current_batch:
            batches.append(current_batch)

        all_results = []
        for batch_idx in tqdm(range(len(batches)), desc="Detecting bboxes",
                              disable=self._disable_tqdm):
            batch_image_idxs = batches[batch_idx]
            batch_images = [images[j].convert("RGB") for j in batch_image_idxs]

            split_index = []
            split_heights = []
            image_splits = []
            for image_idx, image in enumerate(batch_images):
                parts, heights = _split_image(image, self.processor.size["height"])
                image_splits.extend(parts)
                split_index.extend([image_idx] * len(parts))
                split_heights.extend(heights)

            prepared = [self._prepare_image(img) for img in image_splits]
            batch_tensor = torch.stack(prepared, dim=0).to(self._dtype)

            with settings.INFERENCE_MODE():
                pred = self.model(pixel_values=batch_tensor.to(self.model.device))

            logits = pred.logits
            correct_shape = [self.processor.size["height"], self.processor.size["width"]]
            if list(logits.shape[2:]) != correct_shape:
                logits = F.interpolate(logits, size=correct_shape, mode="bilinear",
                                       align_corners=False)

            logits = logits.to(torch.float32).cpu().numpy()
            preds = []
            for i, (idx, height) in enumerate(zip(split_index, split_heights)):
                if len(preds) <= idx:
                    preds.append([logits[i][k] for k in range(heatmap_count)])
                else:
                    heatmaps = preds[idx]
                    pred_heatmaps = [logits[i][k] for k in range(heatmap_count)]
                    if height < self.processor.size["height"]:
                        pred_heatmaps = [h[:height, :] for h in pred_heatmaps]
                    for k in range(heatmap_count):
                        heatmaps[k] = np.vstack([heatmaps[k], pred_heatmaps[k]])
                    preds[idx] = heatmaps

            for pred, j in zip(preds, batch_image_idxs):
                result = parallel_get_boxes(pred, orig_sizes[j], include_maps)
                all_results.append(result)

        return all_results
