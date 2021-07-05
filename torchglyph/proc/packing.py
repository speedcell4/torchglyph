from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import pack_sequence, pack_catted_sequences, batch_sizes_to_ptr, accumulate_sizes

from torchglyph.proc.abc import Proc
from torchglyph.proc.annotations import Device

__all__ = [
    'AsTokenPtr',
    'PackList',
    'PackListList',
]


class AsTokenPtr(Proc):
    def __call__(self, sequence: PackedSequence, **kwargs) -> PackedSequence:
        with torch.no_grad():
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
    def __init__(self, device: Device = None) -> None:
        super(PackList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        if self.device is not None:
            return f'device={self.device}'
        return ''

    def __call__(self, sequences: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(sequences, device=self.device)


class PackListList(Proc):
    def __init__(self, device: Device = None) -> None:
        super(PackListList, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        if self.device is not None:
            return f'device={self.device}'
        return ''

    def __call__(self, sequences: List[Tuple[Tensor, Tensor]], **kwargs) -> PackedSequence:
        return pack_catted_sequences(sequences, device=self.device)
