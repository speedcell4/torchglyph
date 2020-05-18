import functools
from typing import Any
from typing import Union, Tuple, Dict

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch.nn.utils.rnn import pack_padded_sequence


def support_pack(fn):
    @functools.wraps(fn)
    def wrap(x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(x):
            return fn(x, *args, **kwargs)
        else:
            return x._replace(data=fn(x.data, *args, **kwargs))

    return wrap


class SupportPack(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super(SupportPack, self).__init__()
        self.module = module

    def __repr__(self) -> str:
        return f'Packed{self.module.__repr__()}'

    def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        return support_pack(self.module)(x)


class SupportPackMeta(type):
    def __new__(cls, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        forward_fn = attrs.get('forward', bases[0].forward)

        @functools.wraps(forward_fn)
        def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
            if torch.is_tensor(x):
                return forward_fn(self, x, *args, **kwargs)
            else:
                return x._replace(data=forward_fn(self, x.data, *args, **kwargs))

        return type(name, bases, {**attrs, 'forward': forward})


def head_pack(pack: PackedSequence) -> Tensor:
    return pack.data[:pack.batch_sizes[0].item()]


def prepend_pack(pack: PackedSequence, value: Union[int, bool, float, Tensor]) -> PackedSequence:
    if not torch.is_tensor(value):
        value = torch.full_like(head_pack(pack), fill_value=value)
    return pack._replace(
        data=torch.cat([value, pack.data], dim=0),
        batch_sizes=torch.cat([pack.batch_sizes[:1], pack.batch_sizes], dim=0),
    )


def tail_pack(pack: PackedSequence) -> Tensor:
    data, lengths = pad_packed_sequence(pack, batch_first=True)  # type: (Tensor, Tensor)
    indices = torch.arange(lengths.size(0), dtype=torch.long, device=data.device)
    return data[indices, lengths - 1]


def append_pack(pack: PackedSequence, value: Union[int, bool, float, Tensor]) -> PackedSequence:
    if not torch.is_tensor(value):
        value = torch.full_like(head_pack(pack), fill_value=value)
    data, lengths = pad_packed_sequence(pack, batch_first=True)  # type: (Tensor, Tensor)
    indices = torch.arange(lengths.size(0), dtype=torch.long, device=data.device)
    return pack_padded_sequence(
        torch.cat([data, value[:, None]], dim=1).index_put((indices, lengths), value),
        lengths + 1, batch_first=True, enforce_sorted=False,
    )
