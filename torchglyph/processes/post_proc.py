from typing import Any
from typing import Union, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from torchglyph.proc import RecurStrProc, PostProc
from torchglyph.vocab import Vocab


class Numbering(RecurStrProc):
    def process(self, data: str, vocab: Vocab) -> int:
        res = vocab.stoi[data]
        return res


class ToLength(PostProc):
    def __call__(self, ins: List[Any], vocab: Vocab) -> int:
        return len(ins)


class ToMask(PostProc):
    def __init__(self, mask_token: Union[str, int]) -> None:
        super(ToMask, self).__init__()
        self.mask_token = mask_token

    def __call__(self, ins: List[Any], vocab: Vocab) -> List[int]:
        if isinstance(self.mask_token, str):
            assert vocab is not None, 'Vocab is not built yet'
            assert self.mask_token in vocab.stoi, f'{self.mask_token} is not in Vocab'
            mask_idx = vocab.stoi[self.mask_token]
        else:
            mask_idx = self.mask_token
        return [mask_idx for _ in ins]


class ToRange(PostProc):
    def __call__(self, ins: List[Any], vocab: Vocab) -> Any:
        return list(range(len(ins)))


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


class ToPad(PostProc):
    def __init__(self, pad_token: Union[str, int], batch_first: bool = True,
                 dtype: torch.dtype = torch.long) -> None:
        super(ToPad, self).__init__()
        self.dtype = dtype
        self.pad_token = pad_token
        self.batch_first = batch_first

    def __call__(self, ins: List[Tensor], vocab: Vocab) -> None:
        if isinstance(self.pad_token, str):
            assert vocab is not None
            assert self.pad_token in vocab.stoi
            pad_idx = vocab.stoi[self.pad_token]
        else:
            pad_idx = self.pad_token

        return pad_sequence(ins, batch_first=self.batch_first, padding_value=pad_idx)
