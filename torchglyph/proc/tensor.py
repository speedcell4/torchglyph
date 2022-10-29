from typing import List, Union, Set, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number
from torchrua import CattedSequence

from torchglyph.proc.abc import Proc


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype = None, device: Device = None) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype
        self.device = device

    def extra_repr(self) -> str:
        args = []
        if self.dtype is not None:
            args.append(f'dtype={self.dtype}')
        if self.device is not None:
            args.append(f'device={self.device}')
        return ', '.join(args)

    Sequence = Union[Tensor, np.ndarray, Set[Number], List[Number], Tuple[Number, ...]]

    def __call__(self, data: Sequence, **kwargs) -> Tensor:
        if isinstance(data, Tensor):
            tensor = data
        elif isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data)
        return tensor.clone().to(dtype=self.dtype, device=self.device).detach()


class CastAs(Proc):
    def __init__(self, dtype: torch.dtype = None, device: Device = None) -> None:
        super(CastAs, self).__init__()
        self.dtype = dtype
        self.device = device

    def extra_repr(self) -> str:
        args = []
        if self.dtype is not None:
            args.append(f'dtype={self.dtype}')
        if self.device is not None:
            args.append(f'device={self.device}')
        return ', '.join(args)

    Sequence = Union[Tensor, CattedSequence, PackedSequence]

    def __call__(self, data: Sequence, **kwargs) -> Sequence:
        return data.to(dtype=self.dtype, device=self.device)
