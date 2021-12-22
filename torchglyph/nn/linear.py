import torch
from torch import nn

from torchglyph.nn import init

__all__ = [
    'Linear',
]


class Linear(nn.Linear):
    def __init__(self, bias: bool = True, *,
                 in_features: int, out_features: int,
                 dtype: torch.dtype = torch.float32) -> None:
        super(Linear, self).__init__(
            in_features=in_features, out_features=out_features,
            bias=bias, dtype=dtype,
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, fan=self.in_features, a=5 ** 0.5, nonlinearity='leaky_relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)
