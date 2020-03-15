import functools
from typing import Union, Tuple, Dict, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def support_pack(fn):
    @functools.wraps(fn)
    def wrap(x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(x):
            return fn(x, *args, **kwargs)
        else:
            return x._replace(data=fn(x.data, *args, **kwargs))

    return wrap


class SupportPack(type):
    def __new__(cls, name: str, bases: Tuple[type, ...], attrs: Dict[str, Any]):
        forward_fn = bases[0].forward

        @functools.wraps(forward_fn)
        def forward(self, x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
            if torch.is_tensor(x):
                return forward_fn(self, x, *args, **kwargs)
            else:
                return x._replace(data=forward_fn(self, x.data, *args, **kwargs))

        return type(name, bases, {**attrs, 'forward': forward})
