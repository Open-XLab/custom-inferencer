from abc import abstractmethod

from .base import CustomInferencer
from .utils import DetectionResult, ImageType


class DetectionInferencer(CustomInferencer):

    @abstractmethod
    def __call__(self, img: ImageType) -> DetectionResult:
        """This method must be implemented in the child class to define the
        inference process of detection.

        Args:
            img (ImageType): The input image.

        Returns:
            DetectionResult: The outputs of the model.
        """
        pass
