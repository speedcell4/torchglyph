from typing import Union

import torch

from torchglyph.proc import GetRange, ScanL, PackSeq, ToSub, Lift, PadSeq
from torchglyph.pipe import Pipe
from torchglyph.pipe.utilities import cum_index
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


class PaddedTokLengthPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], batch_first: bool = True) -> None:
        super(PaddedTokLengthPipe, self).__init__(
            pre=ToSub() + Lift(GetLength()),
            vocab=None,
            post=ToTensor(),
            batch=PadSeq(pad_token=0, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedTokIndicesPipe(Pipe):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(PackedTokIndicesPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetRange() + ToTensor(),
            batch=ScanL(fn=cum_index, init=0) + PackSeq() + ToDevice(device=device),
        )
