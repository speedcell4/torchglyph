from typing import Union

import torch
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchrua import major_sizes_to_ptr, major_sizes_to_info, CattedSequence

__all__ = [
    'VarDropout',
]


class VarDropout(nn.Dropout):
    def __init__(self, p: float) -> None:
        super(VarDropout, self).__init__(p=p, inplace=True)

    def forward(self, sequence: Union[CattedSequence, PackedSequence]) -> Union[CattedSequence, PackedSequence]:
        if not self.training:
            return sequence

        with torch.no_grad():
            if isinstance(sequence, CattedSequence):
                _, batch_size = major_sizes_to_info(sizes=sequence.token_sizes)
                _, batch_ptr = major_sizes_to_ptr(sizes=sequence.token_sizes.to(device=sequence.data.device))
            elif isinstance(sequence, PackedSequence):
                batch_size, _ = major_sizes_to_info(sizes=sequence.batch_sizes)
                batch_ptr, _ = major_sizes_to_ptr(sizes=sequence.batch_sizes.to(device=sequence.data.device))
            else:
                raise TypeError(f'{type(sequence)} is not supported')

            mask = torch.ones(
                (batch_size, *sequence.data.size()[1:]),
                dtype=sequence.data.dtype, device=sequence.data.device,
            )
            mask = super(VarDropout, self).forward(mask)

        return sequence._replace(data=sequence.data * mask[batch_ptr])
