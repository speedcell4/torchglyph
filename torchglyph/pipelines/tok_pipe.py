from typing import Union

import torch

from torchglyph.dataset import Pipeline
from torchglyph.proc import GetLength, ToDevice
from torchglyph.proc import Numbering, AddToCounter, BuildVocab
from torchglyph.proc import ToTensor


class PaddedTokPipe(Pipeline):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(PaddedTokPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(pad_token=None),
            post_procs=Numbering(),
            batch_procs=ToTensor() + ToDevice(device=device),
        )


class SeqLengthPipe(Pipeline):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(SeqLengthPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=GetLength(),
            batch_procs=ToTensor() + ToDevice(device=device),
        )
