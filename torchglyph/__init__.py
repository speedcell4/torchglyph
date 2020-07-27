from pathlib import Path
from typing import Union

import torch
from torch.nn.utils.rnn import PackedSequence

data_path = (Path.home() / '.torchglyph').expanduser().absolute()
if not data_path.exists():
    data_path.mkdir(parents=True, exist_ok=True)

_BATCH_FIRST = True


def get_batch_first() -> bool:
    return _BATCH_FIRST


def set_batch_first(batch_first: bool) -> None:
    global _BATCH_FIRST
    _BATCH_FIRST = batch_first


def packed_sequence_size(self: PackedSequence, dim: int = None) -> Union[torch.Size, int]:
    if _BATCH_FIRST:
        size = torch.Size((self.batch_sizes[0].item(), self.batch_sizes.size(0), *self.data.size()[1:]))
    else:
        size = torch.Size((self.batch_sizes.size(0), self.batch_sizes[0].item(), *self.data.size()[1:]))
    if dim is None:
        return size
    return size[dim]


def device(self: PackedSequence) -> torch.device:
    return self.data.device


PackedSequence.size = packed_sequence_size
PackedSequence.device = property(device)
del packed_sequence_size, device
