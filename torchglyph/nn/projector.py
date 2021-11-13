import torch
from torch import nn, Tensor

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
