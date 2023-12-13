from typing import List

import torch
from torch import Tensor


def get_shape(shape: torch.Size, dim: int, value: int) -> torch.Size:
    shape = list(shape)
    shape[dim] = value
    return torch.Size(shape)


def broadcast_shapes(*sizes: torch.Size, dim: int) -> List[torch.Size]:
    shape = torch.broadcast_shapes(*[get_shape(size, dim=dim, value=1) for size in sizes])
    return [get_shape(shape, dim=dim, value=size[dim]) for size in sizes]


def broadcast_tensors(*tensors: Tensor, dim: int) -> List[Tensor]:
    shapes = broadcast_shapes(*[tensor.size() for tensor in tensors], dim=dim)
    return [torch.broadcast_to(tensor, shape) for tensor, shape in zip(tensors, shapes)]


def gather(tensor: Tensor, index: Tensor, dim: int) -> Tensor:
    tensor, index = broadcast_tensors(tensor, index, dim=dim)
    return tensor.gather(dim=dim, index=index)
