import torch
from torch import Tensor
from torch.types import Number


def gather(tensor: Tensor, index: Tensor, dim: int) -> Tensor:
    assert tensor.dim() == index.dim(), f'{tensor.dim()} != {index.dim()}'

    tensor_size = torch.broadcast_shapes(tensor.size(), index.size())

    index_size = list(tensor_size)
    index_size[dim] = index.size()[dim]

    tensor = tensor.broadcast_to(tensor_size)
    return tensor.gather(dim=dim, index=index.broadcast_to(index_size))


def mask_fill(tensor: Tensor, *indices: int, value: Number = -float('inf')) -> Tensor:
    for index in indices:
        tensor[..., index] = value

    return tensor
