import torch
from torch import Tensor
from torch.nn.init import calculate_gain

__all__ = [
    'xavier_normal_', 'xavier_uniform_',
    'kaiming_normal_', 'kaiming_uniform_',
]


@torch.no_grad()
def xavier_normal_(tensor: Tensor, fan_in: int, fan_out: int, gain: float = 1.):
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return tensor.normal_(0, std)


@torch.no_grad()
def xavier_uniform_(tensor: Tensor, fan_in: int, fan_out: int, gain: float = 1.):
    bound = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return tensor.uniform_(-bound, +bound)


@torch.no_grad()
def kaiming_normal_(tensor: Tensor, fan: int, a: float = 0, nonlinearity: str = 'leaky_relu'):
    std = calculate_gain(nonlinearity, a) / fan ** 0.5
    return tensor.normal_(0, std)


@torch.no_grad()
def kaiming_uniform_(tensor: Tensor, fan: int, a: float = 0, nonlinearity: str = 'leaky_relu'):
    bound = calculate_gain(nonlinearity, a) * (3.0 / fan) ** 0.5
    return tensor.uniform_(-bound, bound)
