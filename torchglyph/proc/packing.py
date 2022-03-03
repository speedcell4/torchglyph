from abc import ABCMeta
from typing import List, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device
from torchrua import accumulate_sizes, pack_padded_sequence, CattedSequence, PaddedSequence
from torchrua import pack_catted_sequence, pack_sequence
from torchrua import compose_catted_sequences, major_sizes_to_ptr

from torchglyph.proc.abc import Proc

__all__ = [
    'PackingProc',
    'PackSequence',
    'PackCattedSequence',
    'PackPaddedSequence',
    'ToPackedPtrSequence',
    'ComposeCattedSequences',
]


class PackingProc(Proc, metaclass=ABCMeta):
    def __init__(self, device: Device = None) -> None:
        super(PackingProc, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        return f'device={self.device}'

    def __call__(self, data: Any, **kwargs) -> PackedSequence:
        raise NotImplementedError


class PackSequence(PackingProc):
    def __call__(self, data: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(sequences=data, device=self.device)


class PackCattedSequence(PackingProc):
    def __call__(self, data: List[CattedSequence], **kwargs) -> PackedSequence:
        data, token_sizes = data

        if self.device is not None:
            data = data.to(device=self.device)
            token_sizes = token_sizes.to(device=self.device)
        return pack_catted_sequence(sequence=data, token_sizes=token_sizes, device=self.device)


class PackPaddedSequence(PackingProc):
    def __init__(self, batch_first: bool = True, device: Device = None) -> None:
        super(PackPaddedSequence, self).__init__(device=device)
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        return ', '.join([
            f'batch_first={self.batch_first}',
            super(PackPaddedSequence, self).extra_repr()
        ])

    def __call__(self, data: List[PaddedSequence], **kwargs) -> PackedSequence:
        data, token_sizes = data

        return pack_padded_sequence(
            sequence=data.to(device=self.device),
            token_sizes=token_sizes.to(device=self.device),
            batch_first=self.batch_first, device=self.device,
        )


class ToPackedPtrSequence(PackingProc):
    @torch.no_grad()
    def __call__(self, data: PackedSequence, **kwargs) -> PackedSequence:
        device = self.device
        if device is None:
            device = data.data.device

        data = data.data.to(device=device)
        batch_sizes = data.batch_sizes.to(device=device)
        acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

        batch_ptr, _ = major_sizes_to_ptr(batch_sizes=batch_sizes)

        return PackedSequence(
            data=acc_batch_sizes[data] + batch_ptr,
            batch_sizes=data.batch_sizes,
            sorted_indices=data.sorted_indices,
            unsorted_indices=data.unsorted_indices,
        )


class ComposeCattedSequences(PackingProc):
    def __call__(self, data: List[CattedSequence], **kwargs) -> PackedSequence:
        return compose_catted_sequences(sequences=data, device=self.device)
