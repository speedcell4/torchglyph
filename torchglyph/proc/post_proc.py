from typing import Any
from typing import Union, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from torchglyph.proc.abc import Flatten, Proc
from torchglyph.vocab import Vocab


class Numbering(Flatten):
    def process(self, token: str, vocab: Vocab, **kwargs) -> int:
        assert vocab is not None, f'did you forget build_vocab?'

        return vocab.stoi[token]


class ToLength(Proc):
    def __call__(self, ins: List[Any], **kwargs) -> int:
        return len(ins)


class ToMask(Proc):
    def __init__(self, mask_token: Union[str, int]) -> None:
        super(ToMask, self).__init__()
        self.mask_token = mask_token

    def __call__(self, ins: List[Any], vocab: Vocab,**kwargs) -> List[int]:
        if isinstance(self.mask_token, str):
            assert vocab is not None, 'Vocab is not built yet'
            assert self.mask_token in vocab.stoi, f'{self.mask_token} is not in Vocab'
            mask_idx = vocab.stoi[self.mask_token]
        else:
            mask_idx = self.mask_token
        return [mask_idx for _ in ins]


class ToRange(Proc):
    def __call__(self, ins: List[Any],**kwargs) -> Any:
        return list(range(len(ins)))


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def __call__(self, ins: Any,**kwargs) -> Tensor:
        try:
            return torch.tensor(ins, dtype=self.dtype, requires_grad=False)
        except ValueError as err:
            if err.args[0] == "too many dimensions 'str'":
                raise ValueError(f'did you forget {Numbering.__name__} before {ToTensor.__name__}?')
            raise err


class ToPad(Proc):
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
