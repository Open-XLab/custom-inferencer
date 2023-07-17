from abc import abstractmethod

from .base import CustomInferencer
from .utils import ClassificationResult, ImageType


class ImageClassification(CustomInferencer):

    @abstractmethod
    def __call__(self, img: ImageType) -> ClassificationResult:
        """This method must be implemented in the child class to define the
        inference process of image classification.

        Args:
            img (ImageType): The input image.

        Returns:
            ClassificationResult: The outputs of the model.
        """
        pass
