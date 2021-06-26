from typing import Any, Union, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

from torchglyph.proc import Proc

__all__ = [
    'ToTensor', 'ToDevice',
    'Cat', 'Stack',
]


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def extra_repr(self) -> str:
        return f'{self.dtype}'

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
    Data = Union[int, float, bool, Tensor, Tuple[Tensor, ...], PackedSequence]
    Batch = Union[Data, Tuple[Data, ...]]

    def __init__(self, device: Union[int, torch.device]) -> None:
        super(ToDevice, self).__init__()
        if isinstance(device, int):
            if device < 0:
                device = torch.device(f'cpu')
            else:
                device = torch.device(f'cuda:{device}')
        self.device = device

    def extra_repr(self) -> str:
        return f'{self.device}'

    def __call__(self, data: Batch, **kwargs) -> Batch:
        if isinstance(data, (Tensor, PackedSequence)):
            return data.to(device=self.device)
        if isinstance(data, (list, tuple, set)):
            return type(data)([self(datum, **kwargs) for datum in data])
        return data


class Cat(Proc):
    def __init__(self, dim: int) -> None:
        super(Cat, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, data: List[Tensor], **kwargs) -> Tensor:
        return torch.cat(data, dim=self.dim)


class Stack(Proc):
    def __init__(self, dim: int) -> None:
        super(Stack, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, data: List[Tensor], **kwargs) -> Tensor:
        return torch.stack(data, dim=self.dim)
