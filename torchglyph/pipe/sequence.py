from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.pipe.utilities import cum_index
from torchglyph.proc import GetRange, ToDevice, Numbering, UpdateCounter, BuildVocab
from torchglyph.proc import ScanL, ToTensor, PadSeq, PackSeq, StatsVocab, GetMask, ToInt


class RawStrPipe(Pipe):
    def __init__(self) -> None:
        super(RawStrPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=None,
        )


class RawPaddedTensorPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], pad_token: Union[str, int],
                 dtype: torch.dtype = torch.long, batch_first: bool = True) -> None:
        super(RawPaddedTensorPipe, self).__init__(
            pre=ToInt(),
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PadSeq(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class RawPackedTensorPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(RawPackedTensorPipe, self).__init__(
            pre=ToInt(),
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PackSeq() + ToDevice(device=device),
        )


class PaddedSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device],
                 pad_token: Union[str, int], threshold: int = 10, batch_first: bool = True) -> None:
        super(PaddedSeqPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=BuildVocab(special_tokens=(pad_token,)) + StatsVocab(threshold=threshold),
            post=Numbering() + ToTensor(),
            batch=PadSeq(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], threshold: int = 10) -> None:
        super(PackedSeqPipe, self).__init__(
            pre=UpdateCounter(),
            vocab=BuildVocab() + StatsVocab(threshold=threshold),
            post=Numbering() + ToTensor(),
            batch=PackSeq() + ToDevice(device=device),
        )


class PaddedSeqMaskPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], filling_mask: bool) -> None:
        super(PaddedSeqMaskPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetMask(mask_token=1 if filling_mask else 0) + ToTensor(dtype=torch.bool),
            batch=PadSeq(pad_token=0 if filling_mask else 1, batch_first=True) + ToDevice(device=device),
        )


class PackedSeqIndicesPipe(Pipe):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(PackedSeqIndicesPipe, self).__init__(
            pre=None,
            vocab=None,
            post=GetRange() + ToTensor(),
            batch=ScanL(fn=cum_index, init=0) + PackSeq() + ToDevice(device=device),
        )
