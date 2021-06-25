from typing import List, Optional, Union

import torch
from torch import Tensor
from torchrua.padding import pad_sequence

from torchglyph.proc import Proc

__all__ = [
    'PadList',
]


class PadList(Proc):
    def __init__(self, batch_first: bool = True,
                 padding_value: Union[int, bool, float] = 0,
                 device: Optional[torch.device] = None) -> None:
        super(PadList, self).__init__()
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.device = device

    def extra_repr(self) -> str:
        return ', '.join([
            f'batch_first={self.batch_first}',
            f'padding_value={self.padding_value}',
            f'device={self.device}',
        ])

    def __call__(self, sequences: List[Tensor], **kwargs) -> Tensor:
        return pad_sequence(
            sequences=sequences, batch_first=self.batch_first,
            padding_value=self.padding_value, device=self.device,
        )
