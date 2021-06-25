from typing import Optional, List, Tuple

import torch
from torch import Tensor
from torchrua import cat_sequence

from torchglyph.proc import Proc

__all__ = [
    'CatList',
]


class CatList(Proc):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super(CatList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tensor], **kwargs) -> Tuple[Tensor, Tensor]:
        return cat_sequence(sequences=sequences, device=self.device)
