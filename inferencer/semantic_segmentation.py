from abc import abstractmethod

from .base import CustomInferencer
from .utils import ImageType


class SemanticSegmentationInferencer(CustomInferencer):

    @abstractmethod
    def __call__(self, img: ImageType) -> ImageType:
        """This method must be implemented in the child class to define the
        inference process of semantic segmentation.

        Args:
            img (ImageType): The input image.

        Returns:
            ImageType: The outputs of the model.
        """
        pass
