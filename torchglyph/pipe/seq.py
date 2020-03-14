from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import GetLength, Numbering
from torchglyph.proc import ToDevice, UpdateCounter, BuildVocab
from torchglyph.proc import ToTensor, PadSeq, PackSeq, StatsVocab, GetMask


class PaddedRawSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], pad_token: Union[str, int],
                 dtype: torch.dtype = torch.long, batch_first: bool = True) -> None:
        super(PaddedRawSeqPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PadSeq(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PaddedSeqPipe(PaddedRawSeqPipe):
    def __init__(self, device: Union[int, torch.device],
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 8, batch_first: bool = True, dtype: torch.dtype = torch.long) -> None:
        super(PaddedSeqPipe, self).__init__(
            device=device, pad_token=pad_token,
            dtype=dtype, batch_first=batch_first,
        )
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )


class PaddedMaskedSeqPipe(PaddedRawSeqPipe):
    def __init__(self, device: Union[int, torch.device], filling_mask: bool,
                 dtype: torch.dtype = torch.bool, batch_first: bool = True) -> None:
        super(PaddedMaskedSeqPipe, self).__init__(
            device=device, pad_token=(0 if filling_mask else 1),
            dtype=dtype, batch_first=batch_first,
        )
        self.with_(
            post=GetMask(token=1 if filling_mask else 0) + ...,
            batch=PadSeq(pad_token=0 if filling_mask else 1, batch_first=batch_first) + ...,
        )


class PackedRawSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(PackedRawSeqPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PackSeq(enforce_sorted=False) + ToDevice(device=device),
        )


class PackedSeqPipe(PackedRawSeqPipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 8, dtype: torch.dtype = torch.long) -> None:
        super(PackedSeqPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )


class SeqLengthPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(SeqLengthPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetLength(),
            batch=ToTensor(dtype=dtype) + ToDevice(device=device),
        )
