import torch
from torch import Tensor
from torch import nn
from torchrua import RuaMeta

from torchglyph.functional import conjugated_linear
from torchglyph.nn.init import kaiming_uniform_, orthogonal_, bert_normal_, zeros_


class Linear(nn.Linear, metaclass=RuaMeta):
    def __init__(self, bias: bool = True, *, in_features: int, out_features: int) -> None:
        super(Linear, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, fan=self.in_features, a=5 ** 0.5)
        if self.bias is not None:
            zeros_(self.bias)


class KaimingLinear(Linear):
    pass


class OrthogonalLinear(Linear):
    def reset_parameters(self) -> None:
        orthogonal_(self.weight)
        if self.bias is not None:
            zeros_(self.bias)


class BertLinear(Linear):
    def reset_parameters(self) -> None:
        bert_normal_(self.weight)
        if self.bias is not None:
            zeros_(self.bias)


class ConjugatedLinear(nn.Module, metaclass=RuaMeta):
    def __init__(self, num_conjugates: int = 1, bias: bool = True, *, in_features: int, out_features: int) -> None:
        super(ConjugatedLinear, self).__init__()

        self.num_conjugates = num_conjugates
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((num_conjugates, out_features, in_features)))
        self.bias = nn.Parameter(torch.empty((num_conjugates, out_features))) if bias else None

        self.reset_parameters()

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_conjugates={self.num_conjugates}',
            f'in_features={self.in_features}',
            f'out_features={self.out_features}',
            f'bias={self.bias is not None}',
        ])

    def reset_parameters(self) -> None:
        kaiming_uniform_(self.weight, fan=self.in_features, a=5 ** 0.5)
        if self.bias is not None:
            zeros_(self.bias)

    def forward(self, tensor: Tensor) -> Tensor:
        return conjugated_linear(tensor, weight=self.weight, bias=self.bias)


class KaimingConjugatedLinear(ConjugatedLinear):
    pass


class OrthogonalConjugatedLinear(ConjugatedLinear):
    def reset_parameters(self) -> None:
        orthogonal_(self.weight)
        if self.bias is not None:
            zeros_(self.bias)


class BertConjugatedLinear(ConjugatedLinear):
    def reset_parameters(self) -> None:
        bert_normal_(self.weight)
        if self.bias is not None:
            zeros_(self.bias)
