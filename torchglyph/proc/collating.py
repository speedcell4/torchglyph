from typing import List, Union, Set, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device

from torchglyph.proc.abc import Proc

__all__ = [
    'ToTensor',
    'ToDevice',
    'CatTensors',
    'StackTensors',
]


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype = None) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def extra_repr(self) -> str:
        if self.dtype is not None:
            return f'{self.dtype}'
        return ''

    def __call__(self, data: Union[Tensor, np.ndarray, List[int]], **kwargs) -> Tensor:
        try:
            if torch.is_tensor(data):
                tensor = data
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data)
            else:
                tensor = torch.tensor(data)
            return tensor.clone().to(dtype=self.dtype).requires_grad_(False)
        except ValueError:
            raise ValueError(f"'{data}' can not be converted to {Tensor.__name__}")


class ToDevice(Proc):
    Tensors = Union[Tensor, PackedSequence, Set[Tensor], List[Tensor], Tuple[Tensor, ...]]

    def __init__(self, device: Device = None) -> None:
        super(ToDevice, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'{self.device}'

    def __call__(self, tensors: Tensors, **kwargs) -> Tensors:
        if isinstance(tensors, (Tensor, PackedSequence)):
            return tensors.to(device=self.device)
        if isinstance(tensors, (set, list, tuple)):
            return type(tensors)([self(tensor, **kwargs) for tensor in tensors])
        return tensors


class CatTensors(Proc):
    def __init__(self, dim: int) -> None:
        super(CatTensors, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, tensors: List[Tensor], **kwargs) -> Tensor:
        return torch.cat(tensors, dim=self.dim)


class StackTensors(Proc):
    def __init__(self, dim: int) -> None:
        super(StackTensors, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, tensors: List[Tensor], **kwargs) -> Tensor:
        return torch.stack(tensors, dim=self.dim)
