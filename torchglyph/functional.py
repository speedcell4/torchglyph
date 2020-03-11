from typing import Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def packing(fn):
    def wrap(x: Union[Tensor, PackedSequence], *args, **kwargs) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(x):
            return fn(x, *args, **kwargs)
        else:
            return x._replace(data=fn(x.data, *args, **kwargs))

    return wrap
