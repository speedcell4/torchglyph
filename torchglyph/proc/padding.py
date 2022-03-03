from abc import ABCMeta
from typing import List, Any

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number
from torchrua import pad_sequence, pad_packed_sequence, pad_catted_sequence, CattedSequence

from torchglyph.proc.abc import Proc

__all__ = [
    'PaddingProc',
    'PadSequence',
    'PadCattedSequence',
    'PadPackedSequence',
]


class PaddingProc(Proc, metaclass=ABCMeta):
    def __init__(self, batch_first: bool = True,
                 padding_value: Number = 0, device: Device = None) -> None:
        super(PaddingProc, self).__init__()
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.device = device

    def extra_repr(self) -> str:
        return ', '.join([
            f'batch_first={self.batch_first}',
            f'padding_value={self.padding_value}',
            f'device={self.device}',
        ])

    def __call__(self, data: Any, **kwargs) -> Tensor:
        raise NotImplementedError


class PadSequence(PaddingProc):
    def __call__(self, data: List[Tensor], **kwargs) -> Tensor:
        sequence, _ = pad_sequence(
            sequences=data, batch_first=self.batch_first,
            padding_value=self.padding_value, device=self.device,
        )
        return sequence


class PadPackedSequence(PaddingProc):
    def __call__(self, data: PackedSequence, **kwargs) -> Tensor:
        data, _ = pad_packed_sequence(
            sequence=data, batch_first=self.batch_first,
            padding_value=self.padding_value, device=self.device,
        )
        return data


class PadCattedSequence(PaddingProc):
    def __call__(self, data: CattedSequence, **kwargs) -> Tensor:
        sequence, _ = pad_catted_sequence(
            sequence=data, batch_first=self.batch_first,
            padding_value=self.padding_value, device=self.device,
        )
        return sequence
