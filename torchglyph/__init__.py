from pathlib import Path
from typing import Union

import torch
from torch.nn.utils.rnn import PackedSequence

data_path = Path.home() / '.torchglyph'
if not data_path.exists():
    data_path.mkdir(parents=True, exist_ok=True)


def packed_sequence_size(self: PackedSequence, dim: int = None) -> Union[torch.Size, int]:
    size = torch.Size((self.batch_sizes[0].item(), self.batch_sizes.size(0), *self.data.size()[1:]))
    if dim is None:
        return size
    return size[dim]


PackedSequence.size = packed_sequence_size
del packed_sequence_size
