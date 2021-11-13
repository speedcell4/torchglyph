import torch
from torch import Tensor
from torch.nn import init

from torch.nn.init import calculate_gain, constant_, orthogonal_

__all__ = [
    'xavier_normal_', 'xavier_uniform_',
    'kaiming_normal_', 'kaiming_uniform_',
    'bert_normal_',
    'constant_', 'orthogonal_',
]


@torch.no_grad()
def xavier_normal_(tensor: Tensor, fan_in: int, fan_out: int, gain: float = 1.) -> Tensor:
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return init.normal_(tensor, mean=0, std=std)


@torch.no_grad()
def xavier_uniform_(tensor: Tensor, fan_in: int, fan_out: int, gain: float = 1.) -> Tensor:
    bound = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return init.uniform_(tensor, a=-bound, b=+bound)


@torch.no_grad()
def kaiming_normal_(tensor: Tensor, fan: int, a: float = 0., nonlinearity: str = 'leaky_relu') -> Tensor:
    std = calculate_gain(nonlinearity, a) / fan ** 0.5
    return init.normal_(tensor, mean=0, std=std)


@torch.no_grad()
def kaiming_uniform_(tensor: Tensor, fan: int, a: float = 0., nonlinearity: str = 'leaky_relu') -> Tensor:
    bound = calculate_gain(nonlinearity, a) * (3.0 / fan) ** 0.5
    return init.uniform_(tensor, a=-bound, b=+bound)


@torch.no_grad()
def bert_normal_(tensor: Tensor, mean: float = 0., std: float = 0.02) -> Tensor:
    return init.normal_(tensor, mean=mean, std=std)
