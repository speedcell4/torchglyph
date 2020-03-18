from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import Pipe, THRESHOLD
from torchglyph.proc import Lift, ToDevice, Numbering, UpdateCounter, BuildVocab, ToSubList
from torchglyph.proc import ToTensor, PadSeq, PadBlock, PackBlock, StatsVocab


class PaddedIdxBlockPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], pad_token: Optional[str],
                 batch_first: bool = True, dtype: torch.dtype = torch.long) -> None:
        super(PaddedIdxBlockPipe, self).__init__(
            pre=None,
            vocab=None,
            post=Lift(ToTensor(dtype=dtype)) + PadSeq(pad_token=pad_token, batch_first=True),
            batch=PadBlock(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PaddedTokBlockPipe(PaddedIdxBlockPipe):
    def __init__(self, device: Union[int, torch.device],
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 batch_first: bool = True, threshold: int = THRESHOLD, dtype: torch.dtype = torch.long) -> None:
        super(PaddedTokBlockPipe, self).__init__(
            device=device, pad_token=pad_token,
            batch_first=batch_first, dtype=dtype,
        )
        self.with_(
            pre=ToSubList() + Lift(UpdateCounter()),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )


class PackedIdxBlockPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(PackedIdxBlockPipe, self).__init__(
            pre=None,
            vocab=None,
            post=Lift(ToTensor(dtype=dtype)),
            batch=PackBlock(enforce_sorted=False) + ToDevice(device=device),
        )


class PackedTokBlockPipe(PackedIdxBlockPipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = THRESHOLD, dtype: torch.dtype = torch.long) -> None:
        super(PackedTokBlockPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=ToSubList() + Lift(UpdateCounter()),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )
