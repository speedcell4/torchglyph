from typing import List

from torch import Tensor
from torchrua.padding import pad_sequence

from torchglyph.proc.abc import Proc
from torchglyph.proc.annotations import Device, Num

__all__ = [
    'PadList',
]


class PadList(Proc):
    def __init__(self, batch_first: bool = True,
                 padding_value: Num = 0, device: Device = None) -> None:
        super(PadList, self).__init__()
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.device = device

    def extra_repr(self) -> str:
        args = [
            f'batch_first={self.batch_first}',
            f'padding_value={self.padding_value}',
        ]
        if self.device is not None:
            args.append(f'device={self.device}')
        return ', '.join(args)

    def __call__(self, sequences: List[Tensor], **kwargs) -> Tensor:
        return pad_sequence(
            sequences=sequences, batch_first=self.batch_first,
            padding_value=self.padding_value, device=self.device,
        )
