from typing import Any, List

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchglyph.proc.abc import Proc
from torchglyph.annotations import DType, Device, Container

__all__ = [
    'ToTensor', 'ToDevice',
    'Cat', 'Stack',
]


class ToTensor(Proc):
    def __init__(self, dtype: DType = None) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def extra_repr(self) -> str:
        if self.dtype is not None:
            return f'{self.dtype}'
        return ''

    def __call__(self, data: Any, **kwargs) -> Tensor:
        try:
            if torch.is_tensor(data):
                tensor = data.clone()
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).clone()
            else:
                tensor = torch.tensor(data)
            return tensor.to(dtype=self.dtype).requires_grad_(False)
        except ValueError as error:
            if error.args[0] == "too many dimensions 'str'":
                raise ValueError(f"'{data}' can not be converted to {Tensor.__name__}")
            raise error


class ToDevice(Proc):
    def __init__(self, device: Device = None) -> None:
        super(ToDevice, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        if self.device is not None:
            return f'{self.device}'
        return ''

    def __call__(self, data: Container, **kwargs) -> Container:
        if isinstance(data, (Tensor, PackedSequence)):
            return data.to(device=self.device)
        if isinstance(data, (set, list, tuple)):
            return type(data)([self(datum, **kwargs) for datum in data])
        return data


class Cat(Proc):
    def __init__(self, dim: int) -> None:
        super(Cat, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, tensors: List[Tensor], **kwargs) -> Tensor:
        return torch.cat(tensors, dim=self.dim)


class Stack(Proc):
    def __init__(self, dim: int) -> None:
        super(Stack, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, tensors: List[Tensor], **kwargs) -> Tensor:
        return torch.stack(tensors, dim=self.dim)
