from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import PackedTokSeqPipe
from torchglyph.pipe import Pipe
from torchglyph.pipe import THRESHOLD
from torchglyph.proc import GetLength, Lift, ToTensor
from torchglyph.proc.collecting import ToDevice
from torchglyph.proc.contiguous import BuildContiguousSub, BuildContiguousSubPtr, PackContiguousSubPtr


class PackedContiguousSubPipe(PackedTokSeqPipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 seq_token: str, special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = THRESHOLD, dtype: torch.dtype = torch.long) -> None:
        super(PackedContiguousSubPipe, self).__init__(
            device=device, unk_token=unk_token, special_tokens=special_tokens,
            threshold=threshold, dtype=dtype,
        )
        self.with_(
            pre=BuildContiguousSub(seq_token=seq_token) + ...,
        )


class PackedContiguousSubPtrPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(PackedContiguousSubPtrPipe, self).__init__(
            pre=Lift(GetLength()) + BuildContiguousSubPtr() + Lift(ToTensor(dtype=dtype)),
            vocab=None,
            post=None,
            batch=PackContiguousSubPtr(enforce_sorted=False) + ToDevice(device=device),
        )
