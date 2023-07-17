from abc import ABCMeta, abstractmethod
from typing import Optional

import torch
from torch.nn import Module

from .utils import ModelType


class CustomInferencer(metaclass=ABCMeta):
    """Base class for custom inferencers.

    All custom inferencers must inherit from this class.
    """

    def __init__(self,
                 model: ModelType,
                 weights: Optional[str] = None,
                 device: str = 'cpu'):
        """Initialize the custom inferencer.

        Args:
            model (Module | str): The pytorch model to perform inference with.
                If a string is given, it should be the path to a config file
                in OpenMMLab config style.
            weights (str, optional): The path to the weights of the model.
            device (str, optional): The device to perform inference on.
        """

        self.weights = weights
        self.device = device

        if isinstance(model, Module):
            self.model = self._load_pytorch_model(model, weights, device)
        elif isinstance(model, str):
            self.model = self._load_model_from_config(model, weights, device)
        else:
            raise TypeError(f'Invalid model type {type(model)}')

    def _load_pytorch_model(self, model: Module, weights: Optional[str],
                            device: str) -> Module:
        """Load pytorch model to given devices and initialize model
        parameters."""

        if weights is not None:
            model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device)
        model.eval()
        return model

    def _load_model_from_config(self, model: str, weights: Optional[str],
                                device: str) -> Module:
        """Load model and initialize model parameters according from a config
        file and given arguments."""

        raise NotImplementedError

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """This method must be implemented in the child class to define the
        inference process."""
        pass
