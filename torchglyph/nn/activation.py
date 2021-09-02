from typing import Type
from typing import Union

from torch import nn

__all__ = [
    'Activations',
]

Activations = Union[
    Type[nn.Tanh],
    Type[nn.ReLU],
    Type[nn.GELU],
    Type[nn.SELU],
]
