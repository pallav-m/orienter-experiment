# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Modified image processor class for Segformer based on transformers"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from transformers.image_processing_utils import BaseImageProcessor, BatchFeature, get_size_dict
from transformers.image_transforms import to_channel_dimension_format
from transformers.image_utils import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    infer_channel_dimension_format,
    make_list_of_images,
)
from transformers.utils import TensorType

import PIL.Image
import torch

from .s3 import S3DownloaderMixin


class SegformerImageProcessor(S3DownloaderMixin, BaseImageProcessor):
    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        resample: PILImageResampling = PILImageResampling.BILINEAR,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_reduce_labels: bool = False,
        **kwargs,
    ) -> None:
        if "reduce_labels" in kwargs:
            warnings.warn(
                "The `reduce_labels` parameter is deprecated. Use `do_reduce_labels` instead.",
                FutureWarning,
            )
            do_reduce_labels = kwargs.pop("reduce_labels")

        super().__init__(**kwargs)
        size = size if size is not None else {"height": 512, "width": 512}
        size = get_size_dict(size)
        self.do_resize = do_resize
        self.size = size
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else IMAGENET_DEFAULT_MEAN
        self.image_std = image_std if image_std is not None else IMAGENET_DEFAULT_STD
        self.do_reduce_labels = do_reduce_labels
        self._valid_processor_keys = [
            "images", "segmentation_maps", "do_resize", "size", "resample",
            "do_rescale", "rescale_factor", "do_normalize", "image_mean",
            "image_std", "do_reduce_labels", "return_tensors", "data_format",
            "input_data_format",
        ]

    @classmethod
    def from_dict(cls, image_processor_dict: Dict[str, Any], **kwargs):
        image_processor_dict = image_processor_dict.copy()
        if "reduce_labels" in kwargs:
            image_processor_dict["reduce_labels"] = kwargs.pop("reduce_labels")
        return super().from_dict(image_processor_dict, **kwargs)

    def _preprocess(
        self, image, do_resize, do_rescale, do_normalize,
        size=None, resample=None, rescale_factor=None,
        image_mean=None, image_std=None, input_data_format=None,
    ):
        if do_rescale:
            image = self.rescale(image=image, scale=rescale_factor, input_data_format=input_data_format)
        if do_normalize:
            image = self.normalize(image=image, mean=image_mean, std=image_std, input_data_format=input_data_format)
        return image

    def _preprocess_image(
        self, image, do_resize=None, size=None, resample=None,
        do_rescale=None, rescale_factor=None, do_normalize=None,
        image_mean=None, image_std=None, data_format=None, input_data_format=None,
    ) -> np.ndarray:
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(image)
        image = self._preprocess(
            image=image, do_resize=do_resize, size=size, resample=resample,
            do_rescale=do_rescale, rescale_factor=rescale_factor,
            do_normalize=do_normalize, image_mean=image_mean, image_std=image_std,
            input_data_format=input_data_format,
        )
        if data_format is not None:
            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
        return image

    def __call__(self, images, segmentation_maps=None, **kwargs):
        return super().__call__(images, segmentation_maps=segmentation_maps, **kwargs)

    def preprocess(
        self, images: ImageInput, segmentation_maps=None,
        do_resize=None, size=None, resample=None,
        do_rescale=None, rescale_factor=None, do_normalize=None,
        image_mean=None, image_std=None, do_reduce_labels=None,
        return_tensors=None, data_format: ChannelDimension = ChannelDimension.FIRST,
        input_data_format=None, **kwargs,
    ):
        do_resize = do_resize if do_resize is not None else self.do_resize
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        resample = resample if resample is not None else self.resample
        size = size if size is not None else self.size
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        images = make_list_of_images(images)
        images = [
            self._preprocess_image(
                image=img, do_resize=do_resize, resample=resample, size=size,
                do_rescale=do_rescale, rescale_factor=rescale_factor,
                do_normalize=do_normalize, image_mean=image_mean, image_std=image_std,
                data_format=data_format, input_data_format=input_data_format,
            )
            for img in images
        ]
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)
