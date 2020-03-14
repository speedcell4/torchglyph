from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import Lift, ToDevice, Numbering, UpdateCounter, BuildVocab, ToSubList
from torchglyph.proc import ToTensor, PadSeq, PadSub, PackSub, StatsVocab


class PaddedSubPipe(Pipe):
    def __init__(self, device: Union[int, torch.device],
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 batch_first: bool = True, threshold: int = 8, dtype: torch.dtype = torch.long) -> None:
        super(PaddedSubPipe, self).__init__(
            pre=ToSubList() + Lift(UpdateCounter()),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + Lift(ToTensor(dtype=dtype)) + PadSeq(pad_token=pad_token, batch_first=True),
            batch=PadSub(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedSubPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 8, dtype: torch.dtype = torch.long) -> None:
        super(PackedSubPipe, self).__init__(
            pre=ToSubList() + Lift(UpdateCounter()),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + Lift(ToTensor(dtype=dtype)),
            batch=PackSub(enforce_sorted=False) + ToDevice(device=device),
        )
