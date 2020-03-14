from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import GetLength, Numbering
from torchglyph.proc import ToDevice, UpdateCounter, BuildVocab
from torchglyph.proc import ToTensor, PadSeq, PackSeq, StatsVocab, GetMask, RevVocab


class PaddedSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device],
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 8, batch_first: bool = True, dtype: torch.dtype = torch.long) -> None:
        super(PaddedSeqPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ToTensor(dtype=dtype),
            batch=PadSeq(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 8, dtype: torch.dtype = torch.long) -> None:
        super(PackedSeqPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ToTensor(dtype=dtype),
            batch=PackSeq(enforce_sorted=False) + ToDevice(device=device),
        )


class PaddedSeqMaskPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], filling_mask: bool, batch_first: bool = True) -> None:
        super(PaddedSeqMaskPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetMask(token=1 if filling_mask else 0) + ToTensor(dtype=torch.bool),
            batch=PadSeq(pad_token=0 if filling_mask else 1, batch_first=batch_first) + ToDevice(device=device),
        )


class SeqLengthPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(SeqLengthPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetLength(),
            batch=ToTensor(dtype=dtype) + ToDevice(device=device),
        )


class RevStrPipe(Pipe):
    def __init__(self, unk_token: Optional[str], special_tokens: Tuple[Optional[str], ...] = ()) -> None:
        super(RevStrPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
            post=Numbering() + RevVocab(),
            batch=None,
        )
