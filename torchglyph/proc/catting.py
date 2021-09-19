from abc import ABCMeta
from typing import List, Any

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import CattedSequence, cat_sequence, cat_packed_sequence, cat_padded_sequence

from torchglyph.proc.abc import Proc

__all__ = [
    'CattingProc',
    'CatSequence',
    'CatPackedSequence',
    'CatPaddedSequence',
]


class CattingProc(Proc, metaclass=ABCMeta):
    def __init__(self, device: Device = None) -> None:
        super(CattingProc, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, data: Any, **kwargs) -> CattedSequence:
        raise NotImplementedError


class CatSequence(CattingProc):
    def __call__(self, data: List[Tensor], **kwargs) -> CattedSequence:
        return cat_sequence(sequences=data, device=self.device)


class CatPackedSequence(CattingProc):
    def __call__(self, data: PackedSequence, **kwargs) -> CattedSequence:
        return cat_packed_sequence(sequence=data, device=self.device)


class CatPaddedSequence(CattingProc):
    def __init__(self, batch_first: bool = True, device: Device = None) -> None:
        super(CatPaddedSequence, self).__init__(device=device)
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        return ', '.join([
            f'batch_first={self.batch_first}',
            super(CatPaddedSequence, self).extra_repr()
        ])

    def __call__(self, data: PackedSequence, **kwargs) -> CattedSequence:
        data, token_sizes = data
        return cat_padded_sequence(
            sequence=data, token_sizes=token_sizes,
            batch_first=self.batch_first, device=self.device,
        )
