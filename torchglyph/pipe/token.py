from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import GetLength, ToDevice, Numbering, UpdateCounter, BuildVocab
from torchglyph.proc import ToTensor, StatsVocab


class PaddedTokPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Union[str, int], threshold: int = 10) -> None:
        super(PaddedTokPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=BuildVocab(unk_token=unk_token, pad_token=None) + StatsVocab(threshold=threshold),
            post=Numbering(),
            batch=ToTensor() + ToDevice(device=device),
        )


class SeqLengthPipe(Pipe):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(SeqLengthPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetLength(),
            batch=ToTensor() + ToDevice(device=device),
        )
