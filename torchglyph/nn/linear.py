import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from torchglyph.nn import init

__all__ = [
    'Linear', 'CosineLinear',
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


class CosineLinear(Linear):
    def forward(self, tensor: Tensor) -> Tensor:
        tensor = F.normalize(tensor, p=2, dim=-1)
        weight = F.normalize(self.weight, p=2, dim=-1)
        return F.linear(tensor, weight=weight, bias=self.bias)
