import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from torchglyph.nn import init

__all__ = [
    'Projector',
    'CosineProjector',
]


class Projector(nn.Linear):
    def __init__(self, bias: bool = False, *, in_features: int, out_features: int) -> None:
        super(Projector, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, self.in_features, nonlinearity='relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)


class CosineProjector(Projector):
    def __init__(self, bias: bool = False, tau: float = 0.1, *, in_features: int, out_features: int) -> None:
        super(CosineProjector, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
        )
        self.tau = tau

    def extra_repr(self) -> str:
        return ', '.join([
            super(CosineProjector, self).extra_repr(),
            f'tau={self.tau}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        out = torch.cosine_similarity(tensor[..., None, :], self.weight, dim=-1) / self.tau
        if self.bias is not None:
            out = out + self.bias
        return out


class ConjugatedLinear(nn.Module):
    def __init__(self, bias: bool = True, *, in_features: int, num_conjugates: int, out_features: int) -> None:
        super(ConjugatedLinear, self).__init__()

        self.in_features = in_features
        self.num_conjugates = num_conjugates
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((num_conjugates, out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_conjugates, out_features)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, fan=self.in_features, a=5 ** 0.5, nonlinearity='leaky_relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_features={self.in_features}',
            f'num_conjugates={self.num_conjugates}',
            f'out_features={self.out_features}',
            f'bias={self.bias is not None}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        out = (self.weight @ tensor[..., None])[..., 0]
        if self.bias is not None:
            out = out + self.bias
        return out
