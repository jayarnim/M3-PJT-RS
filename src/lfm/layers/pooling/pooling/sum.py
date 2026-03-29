import torch
import torch.nn as nn
from .registry import register


@register("sum")
class ElementwiseSum(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        indices: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        return indices @ embeddings
