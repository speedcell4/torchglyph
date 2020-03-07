from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import GetLength, ToDevice, Numbering, UpdateCounter, BuildVocab
from torchglyph.proc import ToTensor


class PaddedTokPipe(Pipe):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(PaddedTokPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=BuildVocab(pad_token=None),
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
