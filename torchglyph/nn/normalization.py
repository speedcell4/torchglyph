from typing import Type, Union

from torch import Tensor, nn


class Identity(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Identity, self).__init__()

    def reset_parameters(self) -> None:
        pass

    def forward(self, tensor: Tensor, *args, **kwargs) -> Tensor:
        return tensor


class LayerNorm(nn.LayerNorm):
    def __init__(self, *, in_size: int, **kwargs) -> None:
        super(LayerNorm, self).__init__(normalized_shape=in_size)


Normalizations = Union[
    Type[Identity],
    Type[LayerNorm],
]
