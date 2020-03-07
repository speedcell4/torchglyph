from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import GetRange, ToDevice, Numbering, UpdateCounter, BuildVocab, LoadGlove
from torchglyph.proc import ScanL, ToTensor, PadSeq, PackSeq


class RawStrPipe(Pipe):
    def __init__(self) -> None:
        super(RawStrPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=None,
            batch_procs=None,
        )


class RawPaddedTensorPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], pad_token: Union[str, int],
                 dtype: torch.dtype = torch.long, batch_first: bool = True) -> None:
        super(RawPaddedTensorPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToTensor(dtype=dtype),
            batch_procs=PadSeq(pad_token, batch_first) + ToDevice(device=device),
        )


class RawPackedTensorPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(RawPackedTensorPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToTensor(dtype=dtype),
            batch_procs=PackSeq() + ToDevice(device=device),
        )


class PaddedSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device],
                 pad_token: Union[str, int], batch_first: bool = True,
                 dim: int = None) -> None:
        super(PaddedSeqPipe, self).__init__(
            pre_procs=UpdateCounter(),
            vocab_procs=BuildVocab(special_tokens=(pad_token,)) + (LoadGlove('6B', dim) if dim is not None else None),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PadSeq(pad_token, batch_first) + ToDevice(device=device),
        )


class PackedSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(PackedSeqPipe, self).__init__(
            pre_procs=UpdateCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PackSeq() + ToDevice(device=device),
        )


class PackedSeqRangePipe(Pipe):
    def __init__(self, device: Union[int, torch.device]) -> None:
        super(PackedSeqRangePipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=GetRange() + ToTensor(),
            batch_procs=ScanL(lambda t, a: (t + a, t.size(0) + a), 0) + PackSeq() + ToDevice(device=device),
        )
