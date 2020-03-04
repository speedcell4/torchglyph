from typing import List, Any

import torch
from torch import Tensor

from torchglyph.proc import RecurStrProc, PostProc
from torchglyph.vocab import Vocab


class Numbering(RecurStrProc):
    def process(self, data: str, vocab: Vocab) -> int:
        res = vocab.stoi[data]
        return res


class ToTensor(PostProc):
    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def __call__(self, ins: Any, vocab: Vocab) -> Tensor:
        try:
            return torch.tensor(ins, dtype=self.dtype, requires_grad=False)
        except ValueError as err:
            if err.args[0] == "too many dimensions 'str'":
                raise ValueError(f'did you forget {Numbering.__name__} before {ToTensor.__name__}?')
            raise err


class ToTensorList(PostProc):
    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(ToTensorList, self).__init__()
        self.process = ToTensor(dtype=dtype)

    def __call__(self, ins: List[Any], vocab: Vocab) -> List[Tensor]:
        return [self.process(d, vocab=vocab) for d in ins]
