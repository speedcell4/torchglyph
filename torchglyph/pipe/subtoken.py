from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import Lift, ToDevice, Numbering, UpdateCounter, BuildVocab
from torchglyph.proc import ToSub, ToTensor, PadSeq, PadSub, PackSub, StatsVocab


class PaddedSubPipe(Pipe):
    def __init__(self, device: Union[int, torch.device],
                 unk_token: Union[str, int], pad_token: Union[str, int], threshold: int = 10) -> None:
        super(PaddedSubPipe, self).__init__(
            pre=ToSub() + Lift(UpdateCounter()),
            vocab=BuildVocab(unk_token=unk_token, pad_token=pad_token) + StatsVocab(threshold=threshold),
            post=Numbering() + Lift(ToTensor()) + PadSeq(pad_token=pad_token, batch_first=True),
            batch=PadSub(pad_token=pad_token) + ToDevice(device=device),
        )


class PackedSubPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Union[str, int], threshold: int = 10) -> None:
        super(PackedSubPipe, self).__init__(
            pre=ToSub() + Lift(UpdateCounter()),
            vocab=BuildVocab(unk_token=unk_token, pad_token=None) + StatsVocab(threshold=threshold),
            post=Numbering() + Lift(ToTensor()),
            batch=PackSub() + ToDevice(device=device),
        )
