from typing import List, Tuple, Optional

import torch
from torch.types import Device, Number
from torchrua import CattedSequence

from torchglyph.pipe.abc import Pipe
from torchglyph.proc.catting import CatSequence
from torchglyph.proc.tensor import ToTensor
from torchglyph.proc.vocab import CountTokenSequence, BuildVocab, StatsVocab, ToIndexSequence

__all__ = [
    'CattedNumListPipe',
    'CattedStrListPipe',
]


class CattedNumListPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(CattedNumListPipe, self).__init__(
            pre=None,
            post=ToTensor(dtype=dtype),
            vocab=None,
            batch=CatSequence(device=device),
        )

    def inv(self, sequence: CattedSequence) -> List[List[Number]]:
        data = sequence.data.detach()
        token_sizes = sequence.token_sizes.detach().cpu().tolist()

        return [
            tensor.detach().tolist()
            for tensor in torch.split(data, token_sizes, dim=0)
        ]


class CattedStrListPipe(CattedNumListPipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long,
                 unk_token: Optional[str] = '<unk>',
                 special_tokens: Tuple[Optional[str], ...] = (), threshold: int = 10) -> None:
        super(CattedStrListPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=CountTokenSequence(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=ToIndexSequence() + ...,
        )

    def inv(self, sequence: CattedSequence) -> List[List[str]]:
        assert sequence.data.dim() == 1, f'{sequence.data.dim()} != 1'

        return [
            [self.vocab.inv(index) for index in indices]
            for indices in super(CattedStrListPipe, self).inv(sequence)
        ]
