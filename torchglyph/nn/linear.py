import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from torchglyph.nn import init

__all__ = [
    'Linear', 'ConjugatedLinear',
    'Projector', 'ConjugatedProjector',
]


class Linear(nn.Linear):
    def __init__(self, bias: bool = True, normalize: bool = False, *,
                 in_features: int, out_features: int,
                 dtype: torch.dtype = torch.float32) -> None:
        super(Linear, self).__init__(
            in_features=in_features, out_features=out_features,
            bias=bias, dtype=dtype,
        )
        self.normalize = normalize

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, fan=self.in_features, a=5. ** 0.5, nonlinearity='leaky_relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.normalize:
            cls_name = f'Cos{cls_name}'
        return f'{cls_name}({self.extra_repr()})'

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.normalize:
            weight = self.weight
        else:
            tensor = F.normalize(tensor, p=2, dim=-1)
            weight = F.normalize(self.weight, p=2, dim=-1)

        return F.linear(tensor, weight=weight, bias=self.bias)


class ConjugatedLinear(nn.Module):
    def __init__(self, bias: bool = True, normalize: bool = False, *,
                 in_features: int, num_conjugates: int, out_features: int,
                 dtype: torch.dtype = torch.float32) -> None:
        super(ConjugatedLinear, self).__init__()

        self.weight = nn.Parameter(torch.empty((num_conjugates, out_features, in_features), dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.empty((num_conjugates, out_features), dtype=dtype))
        else:
            self.register_parameter('bias', None)

        self.in_features = in_features
        self.num_conjugates = num_conjugates
        self.out_features = out_features
        self.normalize = normalize

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, fan=self.in_features, a=5. ** 0.5, nonlinearity='leaky_relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)

    def __repr__(self) -> str:
        cls_name = self.__class__.__name__
        if self.normalize:
            cls_name = f'Cos{cls_name}'
        return f'{cls_name}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'in_features={self.in_features}',
            f'num_conjugates={self.num_conjugates}',
            f'out_features={self.out_features}',
            f'bias={self.bias is not None}',
        ])

    def forward(self, tensor: Tensor) -> Tensor:
        if not self.normalize:
            weight = self.weight
        else:
            tensor = F.normalize(tensor, p=2, dim=-1)
            weight = F.normalize(self.weight, p=2, dim=-1)

        out = (weight @ tensor[..., None])[..., 0]
        if self.bias is not None:
            out = out + self.bias
        return out


class Projector(Linear):
    def __init__(self, normalize: bool = False, *,
                 in_features: int, out_features: int,
                 dtype: torch.dtype = torch.float32) -> None:
        super(Projector, self).__init__(
            in_features=in_features, out_features=out_features,
            bias=False, normalize=normalize, dtype=dtype,
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, fan=self.in_features, a=0., nonlinearity='relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)


class ConjugatedProjector(ConjugatedLinear):
    def __init__(self, normalize: bool = False, *,
                 in_features: int, num_conjugates: int, out_features: int,
                 dtype: torch.dtype = torch.float32) -> None:
        super(ConjugatedProjector, self).__init__(
            in_features=in_features, num_conjugates=num_conjugates, out_features=out_features,
            bias=False, normalize=normalize, dtype=dtype,
        )

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, fan=self.in_features, a=0., nonlinearity='relu')
        if self.bias is not None:
            init.constant_(self.bias, 0.)
