from . import pooling
from .pooling.base import PoolingLayer
from .pooling.registry import POOLING_REGISTRY


def build(
    name: str, 
    **kwargs,
) -> PoolingLayer:
    cls = POOLING_REGISTRY[name]
    return cls(**kwargs)