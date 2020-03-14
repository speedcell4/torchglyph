from typing import Union, Optional, Tuple

import torch

from torchglyph.pipe import Pipe
from torchglyph.pipe.utilities import THRESHOLD
from torchglyph.proc import GetLength, ToDevice, Numbering, UpdateCounter, BuildVocab
from torchglyph.proc import ToTensor, StatsVocab


class RawTokTensorPipe(Pipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(RawTokTensorPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=ToTensor(dtype=dtype) + ToDevice(device=device),
        )


class TokTensorPipe(RawTokTensorPipe):
    def __init__(self, device: Union[int, torch.device], unk_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = THRESHOLD, dtype: torch.dtype = torch.long) -> None:
        super(TokTensorPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering(),
        )


class SeqLengthTensorPipe(RawTokTensorPipe):
    def __init__(self, device: Union[int, torch.device], dtype: torch.dtype = torch.long) -> None:
        super(SeqLengthTensorPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=GetLength(),
        )
