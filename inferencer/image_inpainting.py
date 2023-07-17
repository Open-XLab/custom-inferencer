from abc import abstractmethod

from .base import CustomInferencer
from .utils import ImageType


class ImageInpaintingInferencer(CustomInferencer):

    @abstractmethod
    def __call__(self, img: ImageType, mask: ImageType) -> ImageType:
        """This method must be implemented in the child class to define the
        inference process of image inpainting.

        Args:
            img (ImageType): The input image.
            msak (ImageType): The inpainting mask of the input image.

        Returns:
            ImageType: The outputs of the model.
        """
        pass
