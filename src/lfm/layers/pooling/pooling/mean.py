import torch
import torch.nn as nn
from .registry import register


@register("mean")
class ElementwiseMean(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, 
        indices: torch.Tensor,
        embeddings: torch.Tensor,
    ):
        slice = indices @ embeddings
        count = indices.sum(dim=1, keepdim=True).clamp(min=1)
        return slice / count