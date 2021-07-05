from typing import List

from torch import Tensor
from torchrua import cat_sequence

from torchglyph.proc.abc import Proc
from torchglyph.proc.annotations import Device, CattedSequence

__all__ = [
    'CatList',
]


class CatList(Proc):
    def __init__(self, device: Device = None) -> None:
        super(CatList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tensor], **kwargs) -> CattedSequence:
        return cat_sequence(sequences=sequences, device=self.device)
