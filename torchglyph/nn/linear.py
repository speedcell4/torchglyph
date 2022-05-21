import torch
from torch import Tensor
from torch import nn
from torch.nn.init import uniform_, normal_, constant_

from torchglyph.functional import conjugated_linear
from torchglyph.nn.init import kaiming_uniform_


class Linear(nn.Module):
    def __init__(self, num_conjugates: int, in_features: int, out_features: int, bias: bool = True) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_conjugates = num_conjugates

        self.weight = nn.Parameter(torch.empty((num_conjugates, out_features, in_features)))

        if bias:
            self.bias = nn.Parameter(torch.empty((num_conjugates, out_features)))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, fan=self.in_features, a=5 ** 0.5)
        if self.bias is not None:
            bound = self.in_features ** -0.5
            uniform_(self.bias, -bound, +bound)

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_features}',
            f'{self.out_features}',
            f'num_conjugates={self.num_conjugates}',
            f'bias={self.bias is not None}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        return conjugated_linear(tensor, weight=self.weight, bias=self.bias)


class TransformerLinear(Linear):
    @torch.no_grad()
    def reset_parameters(self) -> None:
        normal_(self.weight, mean=0, std=0.05)
        if self.bias is not None:
            constant_(self.bias, 0.)
