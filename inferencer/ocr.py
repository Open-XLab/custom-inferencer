from abc import abstractmethod

from .base import CustomInferencer
from .utils import ImageType, OCRResult


class OCRInferencer(CustomInferencer):

    @abstractmethod
    def __call__(self, img: ImageType) -> OCRResult:
        """This method must be implemented in the child class to define the
        inference process of OCR.

        Args:
            img (ImageType): The input image.

        Returns:
            OCR: The outputs of the model.
        """
        pass
