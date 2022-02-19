from typing import Tuple, List

import torch
from torch import Tensor
from torch.types import Device, Number

from torchglyph.pipe.abc import Pipe
from torchglyph.proc.collating import ToTensor, ToDevice
from torchglyph.proc.padding import PadSequence
from torchglyph.proc.vocab import UpdateCounter, BuildVocab, StatsVocab, Numbering

__all__ = [
    'PaddedNumPipe', 'PaddedNumListPipe',
    'PaddedStrPipe', 'PaddedStrListPipe',
]


class PaddedNumPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(PaddedNumPipe, self).__init__(
            pre=None,
            vocab=None,
            post=None,
            batch=ToTensor(dtype=dtype) + ToDevice(device=device),
        )

    def inv(self, data: Tensor) -> List[Number]:
        return data.detach().cpu().tolist()


class PaddedStrPipe(PaddedNumPipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long,
                 unk_token: str = '<unk>', pad_token: str = '<pad>',
                 special_tokens: Tuple[str, ...] = (), threshold: int = 10) -> None:
        super(PaddedStrPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
            batch=...,
        )

    def inv(self, data: Tensor) -> List[str]:
        return [
            self.vocab.inv(index)
            for index in super(PaddedStrPipe, self).inv(data=data)
        ]


class PaddedNumListPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long,
                 batch_first: bool = True, padding_value: Number = 0) -> None:
        super(PaddedNumListPipe, self).__init__(
            post=ToTensor(dtype=dtype),
            batch=PadSequence(batch_first=batch_first, padding_value=padding_value, device=device),
        )

    def inv(self, data: Tensor, token_sizes: Tensor) -> List[List[Number]]:
        data = data.detach().cpu().tolist()
        token_sizes = token_sizes.detach().cpu().tolist()

        return [
            [data[index1][index2] for index2 in range(token_size)]
            for index1, token_size in enumerate(token_sizes)
        ]


class PaddedStrListPipe(PaddedNumListPipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long, batch_first: bool = True,
                 unk_token: str = '<unk>', pad_token: str = '<pad>',
                 special_tokens: Tuple[str, ...] = (), threshold: int = 10) -> None:
        super(PaddedStrListPipe, self).__init__(
            batch_first=batch_first,
            padding_value=0,  # TODO: fix padding_value
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
            [self.vocab.inv(index) for index in indices]
            for indices in super(PaddedStrListPipe, self).inv(data=data, token_sizes=token_sizes)
        ]
