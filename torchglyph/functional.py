import torch
from torch import Tensor


def conjugated_linear(tensor: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    out = torch.einsum('cyx,...cx->...cy', weight, tensor)
    if bias is not None:
        out = out + bias
    return out
