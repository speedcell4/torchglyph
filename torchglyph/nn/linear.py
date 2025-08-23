from typing import Tuple, Type, Union

import torch
from torch import Tensor, nn
from torch.nn import init


class Linear(nn.Module):
    def __init__(self, bias: bool = True, *, in_features: int, out_features: int,
                 leading_features: Tuple[int, ...] = ()) -> None:
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.leading_features = leading_features

        self.weight = nn.Parameter(torch.empty((*leading_features, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((*leading_features, out_features))) if bias else None

        self.reset_parameters()

    def extra_repr(self) -> str:
        args = [
            f'in_features={self.in_features}',
            f'out_features={self.out_features}',
            f'bias={self.bias is not None}',
        ]
        if self.leading_features != ():
            args.append(f'leading_features={self.leading_features}')

        return ', '.join(args)

    def reset_parameters(self) -> None:
        bound = self.in_features ** -0.5
        init.uniform_(self.weight, -bound, +bound)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = torch.einsum('...x,...yx->...y', tensor, self.weight)
        if self.bias is not None:
            tensor = tensor + self.bias

        return tensor


class NormLinear(Linear):
    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            init.zeros_(self.bias)


class OrthLinear(Linear):
    def reset_parameters(self) -> None:
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


class ZeroLinear(Linear):
    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


class Bilinear(nn.Module):
    def __init__(self, bias: bool = True, *, in_features1: int, in_features2: int, out_features: int,
                 leading_features: Tuple[int, ...] = ()) -> None:
        super(Bilinear, self).__init__()

        self.in_features1 = in_features1
        self.in_features2 = in_features2
        self.out_features = out_features
        self.leading_features = leading_features

        self.weight = nn.Parameter(torch.empty((*leading_features, out_features, in_features2, in_features1)))
        self.bias = nn.Parameter(torch.empty((*leading_features, out_features))) if bias else None

        self.reset_parameters()

    def extra_repr(self) -> str:
        args = [
            f'in_features1={self.in_features1}',
            f'in_features2={self.in_features2}',
            f'out_features={self.out_features}',
            f'bias={self.bias is not None}',
        ]
        if self.leading_features != ():
            args.append(f'leading_features={self.leading_features}')

        return ', '.join(args)

    def reset_parameters(self) -> None:
        bound = max(self.in_features1, self.in_features2) ** -0.5
        init.uniform_(self.weight, -bound, +bound)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, tensor1: Tensor, tensor2: Tensor) -> Tensor:
        tensor = torch.einsum('...x,...zyx,...y->...z', tensor1, self.weight, tensor2)
        if self.bias is not None:
            tensor = tensor + self.bias

        return tensor


class NormBilinear(Bilinear):
    def reset_parameters(self) -> None:
        init.normal_(self.weight, mean=0.0, std=0.02)
        if self.bias is not None:
            init.zeros_(self.bias)


class OrthBilinear(Bilinear):
    def reset_parameters(self) -> None:
        init.orthogonal_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


class ZeroBilinear(Bilinear):
    def reset_parameters(self) -> None:
        init.zeros_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)


Proj = Union[
    Type[Linear],
    Type[NormLinear],
    Type[OrthLinear],
    Type[ZeroLinear],
]

Biproj = Union[
    Type[Bilinear],
    Type[NormBilinear],
    Type[OrthBilinear],
    Type[ZeroBilinear],
]
