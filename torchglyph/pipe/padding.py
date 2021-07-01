from typing import Union, Optional, Tuple, List

import torch
from torch import Tensor

from torchglyph.pipe.abc import Pipe, THRESHOLD
from torchglyph.proc.list import ToTokenSize
from torchglyph.proc.padding import PadList
from torchglyph.proc.tensor import ToTensor, ToDevice
from torchglyph.proc.vocab import UpdateCounter, BuildVocab, StatsVocab, Numbering

__all__ = [
    'TokenSizesPipe',
    'PadListNumPipe', 'PadListStrPipe',
]


class TokenSizesPipe(Pipe):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.long) -> None:
        super(TokenSizesPipe, self).__init__(
            pre=ToTokenSize(),
            vocab=None,
            post=None,
            batch=ToTensor(dtype=dtype) + ToDevice(device=device),
        )


class PadListNumPipe(Pipe):
    def __init__(self, batch_first: bool, padding_value: Union[int, bool, float],
                 device: torch.device, dtype: torch.dtype = torch.long) -> None:
        super(PadListNumPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PadList(batch_first=batch_first, padding_value=padding_value, device=device),
        )

    def inv(self, data: Tensor, token_sizes: Tensor) -> List[List[Tuple[int, bool, float]]]:
        data = data.detach().cpu().tolist()
        token_sizes = token_sizes.detach().cpu().tolist()

        return [
            [data[index1][index2] for index2 in range(token_size)]
            for index1, token_size in enumerate(token_sizes)
        ]


class PadListStrPipe(PadListNumPipe):
    def __init__(self, batch_first: bool, padding_value: Union[int, bool, float], device: torch.device,
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = THRESHOLD, dtype: torch.dtype = torch.long) -> None:
        super(PadListStrPipe, self).__init__(
            batch_first=batch_first, padding_value=padding_value,
            device=device, dtype=dtype,
        )
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )

    def inv(self, data: Tensor, token_sizes: Tensor) -> List[List[str]]:
        assert data.dim() == 2, f'{data.dim()} != 2'
        assert token_sizes.dim() == 1, f'{token_sizes.dim()} == {1}'

        return [
            [self.vocab.itos[datum] for datum in data]
            for data in super(PadListStrPipe, self).inv(data=data, token_sizes=token_sizes)
        ]
