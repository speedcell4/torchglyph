from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import Pipe
from torchglyph.pipe.utilities import cum_index
from torchglyph.proc import GetLength, ToDevice, Numbering, UpdateCounter, BuildVocab, ToSubList
from torchglyph.proc import GetRange, ScanL, PackSeq, Lift, PadSeq
from torchglyph.proc import ToTensor, StatsVocab


class PaddedTokPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 8, dtype: torch.dtype = torch.long) -> None:
        super(PaddedTokPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering(),
            batch=ToTensor(dtype=dtype) + ToDevice(device=device),
        )


class TokLengthPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], batch_first: bool = True,
                 dtype: torch.dtype = torch.long) -> None:
        super(TokLengthPipe, self).__init__(
            pre=ToSubList() + Lift(GetLength()),
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PadSeq(pad_token=0, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedTokIndicesPipe(Pipe):
    def __init__(self, device: Union[int, torch.device],
                 reverse: bool = False, dtype: torch.dtype = torch.long) -> None:
        super(PackedTokIndicesPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetRange(reverse=reverse) + ToTensor(dtype=dtype),
            batch=ScanL(fn=cum_index, init=0) + PackSeq(enforce_sorted=False) + ToDevice(device=device),
        )
