from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import ToTensor, PadSeq, ToDevice, PackSeq


class RawStrPipe(Pipe):
    def __init__(self) -> None:
        super(RawStrPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=None,
        )


class PaddedRawSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], pad_token: Union[str, int],
                 dtype: torch.dtype = torch.long, batch_first: bool = True) -> None:
        super(PaddedRawSeqPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PadSeq(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedRawSeqPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(PackedRawSeqPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PackSeq(enforce_sorted=False) + ToDevice(device=device),
        )
