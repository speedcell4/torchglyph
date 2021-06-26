from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import pack_sequence, reduce_catted_sequences, batch_sizes_to_ptr, accumulate_sizes

from torchglyph.proc import Proc

__all__ = [
    'ToTokenPtr',
    'PackList',
    'PackListList',
]


class ToTokenPtr(Proc):
    def __call__(self, sequence: PackedSequence, **kwargs) -> PackedSequence:
        batch_sizes = sequence.batch_sizes.to(device=sequence.data.device)
        _, batch_ptr, _ = batch_sizes_to_ptr(batch_sizes=batch_sizes)
        acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

        return PackedSequence(
            data=acc_batch_sizes[sequence.data] + batch_ptr,
            batch_sizes=sequence.batch_sizes,
            sorted_indices=sequence.sorted_indices,
            unsorted_indices=sequence.unsorted_indices,
        )


class PackList(Proc):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super(PackList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(sequences, device=self.device)


class PackListList(Proc):
    def __init__(self, device: Optional[torch.device] = None) -> None:
        super(PackListList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, sequences: List[Tuple[Tensor, Tensor]], **kwargs) -> PackedSequence:
        return reduce_catted_sequences(sequences, device=self.device)
