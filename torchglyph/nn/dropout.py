from typing import Union

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchrua import CattedSequence, RuaMeta
from torchrua.utils import sequence_ptr, sequence_shape

Sequence = Union[CattedSequence, PackedSequence]


class Dropout(nn.Dropout, metaclass=RuaMeta):
    def __init__(self, p: float = 0.0) -> None:
        super(Dropout, self).__init__(p=p, inplace=False)


class VarDropout(nn.Dropout):
    def __init__(self, p: float = 0.0) -> None:
        super(VarDropout, self).__init__(p=p, inplace=True)

    def forward(self, sequence: Sequence) -> Sequence:
        if not self.training:
            return sequence

        _, batch_size, _ = sequence_shape(sequence=sequence)
        batch_ptr, _ = sequence_ptr(sequence=sequence)

        mask = torch.ones(
            (batch_size, *sequence.data.size()[1:]),
            dtype=sequence.data.dtype, device=sequence.data.device,
        )
        mask = super(VarDropout, self).forward(mask)

        return sequence._replace(data=sequence.data * mask[batch_ptr])
