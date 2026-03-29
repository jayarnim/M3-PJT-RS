from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class PoolingLayer(nn.Module, ABC):
    @abstractmethod
    def forward(
        self, 
        *args,
    ) -> torch.Tensor:
        raise NotImplementedError