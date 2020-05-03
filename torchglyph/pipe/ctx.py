from typing import Union

import torch

from torchglyph.pipe import Pipe
from torchglyph.proc import ToDevice
from torchglyph.proc.ctx import PadELMo
from torchglyph.proc.tokenizer import ELMoTokenizer


class ELMoPipe(Pipe):
    def __init__(self, device: Union[int, torch.device]):
        super(ELMoPipe, self).__init__(
            pre=ELMoTokenizer(),
            vocab=None,
            post=None,
            batch=PadELMo() + ToDevice(device=device),
        )
