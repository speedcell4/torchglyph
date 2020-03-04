from typing import List, Any

import torch
from torch import Tensor

from torchglyph.proc import RecurStrProc, PostProc
from torchglyph.vocab import Vocab


class Numbering(RecurStrProc):
    def process(self, data: str, vocab: Vocab) -> int:
        return vocab.stoi[data]


class ToTensor(PostProc):
    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def __call__(self, ins: Any, vocab: Vocab) -> Tensor:
        return torch.tensor(ins, dtype=self.dtype, requires_grad=False)


class ToTensorList(PostProc):
    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(ToTensorList, self).__init__()
        self.dtype = dtype

    def __call__(self, ins: List[Any], vocab: Vocab) -> List[Tensor]:
        return [torch.tensor(d, dtype=self.dtype, requires_grad=False) for d in ins]