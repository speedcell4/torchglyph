from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import pack_sequence, reduce_catted_sequences

from torchglyph.proc import Proc

__all__ = [
    'PackList',
    'PackListList',
]


class PackList(Proc):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super(PackList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(sequences, device=self.device)


class PackListList(Proc):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super(PackListList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tuple[Tensor, Tensor]], **kwargs) -> PackedSequence:
        return reduce_catted_sequences(sequences, device=self.device)
