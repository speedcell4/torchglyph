from typing import Union

import torch

from torchglyph.dataset import Pipeline
from torchglyph.proc import Lift, ToDevice
from torchglyph.proc import Numbering, AddToCounter, BuildVocab
from torchglyph.proc import ToSub
from torchglyph.proc import ToTensor, PadSeq, PadSub, PackSub


class PaddedSubPipe(Pipeline):
    def __init__(self, device: Union[int, torch.device],
                 pad_token: Union[int, str] = '<pad>',
                 batch_first: bool = True) -> None:
        super(PaddedSubPipe, self).__init__(
            pre_procs=ToSub() + AddToCounter(),
            vocab_procs=BuildVocab(pad_token=pad_token),
            post_procs=Numbering() + Lift(ToTensor()) + PadSeq(pad_token=pad_token, batch_first=True),
            batch_procs=PadSub(pad_token=pad_token, batch_first=batch_first) + ToDevice(device=device),
        )


class PackedSubPipe(Pipeline):
    def __init__(self, device: Union[int, torch.device], ) -> None:
        super(PackedSubPipe, self).__init__(
            pre_procs=ToSub() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + Lift(ToTensor()),
            batch_procs=PackSub() + ToDevice(device=device),
        )
