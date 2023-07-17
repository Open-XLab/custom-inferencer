from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from PIL.Image import Image
from torch.nn import Module

# Define type aliases
ModelType = Union[Module, str]
ImageType = Image


@dataclass
class ClassificationResult:
    """The result of image classification."""
    # The predicted label (category index)
    label: int
    # The confidence score of the prediction
    score: Optional[float]
    # The predicted category name
    category: Optional[str]
    # The scores of all categories
    full_scores: Optional[np.ndarray]


@dataclass
class InstanceMask:
    """The dataclass of the instance segmentation mask."""
    size: list[int]  # The size of the mask
    counts: str  # The RLE-encoded mask


@dataclass
class InstancePose:
    """The dataclass of the pose (keypoint) information of an object."""
    keypoints: list[list[float]]  # The coordinates of all keypoints
    keypoint_scores: Optional[list[float]]  # The scores of all keypoints


@dataclass
class DetectionResult:
    """The result of object detection."""
    # The bounding boxes of all detected objects
    bboxes: Optional[list[list[int]]]
    # The labels of all detected objects
    labels: Optional[list[int]]
    # The scores of all detected objects
    scores: Optional[list[float]]
    # The segmentation masks of all detected objects
    masks: Optional[list[InstanceMask]]
    # The pose estimation results of all detected objects
    poses: Optional[list[InstancePose]]


@dataclass
class OCRResult:
    """The result of OCR."""
    # The recognized texts
    rec_texts: Optional[list[str]]
    # The confidence scores of the recognized texts
    rec_scores: Optional[list[float]]
    # The polygon coordinates of the texts
    det_polygons: Optional[list[list[float]]]
    # The confidence scores of the detected polygons
    det_scores: Optional[list[float]]
