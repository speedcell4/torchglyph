from typing import List, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import pack_sequence, pack_catted_sequences, batch_sizes_to_ptr, accumulate_sizes, pack_padded_sequence
from torchrua import pack_catted_sequence

from torchglyph.proc.abc import Proc
from torchglyph.proc.annotations import Device, CattedSequence, PaddedSequence

__all__ = [
    'PackProc',
    'PackSequences', 'PackCattedSequence', 'PackPaddedSequence',
    'AsTokenPtr', 'ReduceCattedSequences',
]


class PackProc(Proc):
    def __init__(self, device: Device = None) -> None:
        super(PackProc, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        if self.device is not None:
            return f'device={self.device}'
        return ''

    def __call__(self, sequence: Any, **kwargs) -> PackedSequence:
        raise NotImplementedError


class AsTokenPtr(PackProc):
    def __call__(self, sequence: PackedSequence, **kwargs) -> PackedSequence:
        with torch.no_grad():
            if self.device is None:
                device = sequence.data.device
            else:
                device = self.device

            batch_sizes = sequence.batch_sizes.to(device=device)
            acc_batch_sizes = accumulate_sizes(sizes=batch_sizes)

            _, batch_ptr, _ = batch_sizes_to_ptr(batch_sizes=batch_sizes)

        return PackedSequence(
            data=acc_batch_sizes[sequence.data.to(device=device)] + batch_ptr,
            batch_sizes=sequence.batch_sizes,
            sorted_indices=sequence.sorted_indices,
            unsorted_indices=sequence.unsorted_indices,
        )


class PackSequences(PackProc):
    def __call__(self, sequences: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(sequences=sequences, device=self.device)


class PackCattedSequence(PackProc):
    def __call__(self, sequence: List[CattedSequence], **kwargs) -> PackedSequence:
        sequence, token_sizes = sequence

        if self.device is not None:
            sequence = sequence.to(device=self.device)
            token_sizes = token_sizes.to(device=self.device)
        return pack_catted_sequence(sequence=sequence, token_sizes=token_sizes)


class PackPaddedSequence(PackProc):
    def __init__(self, batch_first: bool = True, device: Device = None) -> None:
        super(PackPaddedSequence, self).__init__(device=device)
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        args = [
            f'batch_first={self.batch_first}'
        ]
        super_args = super(PackPaddedSequence, self).extra_repr()
        if super_args != '':
            args.append(super_args)
        return ', '.join(args)

    def __call__(self, sequence: List[PaddedSequence], **kwargs) -> PackedSequence:
        sequence, token_sizes = sequence

        if self.device is not None:
            sequence = sequence.to(device=self.device)
            token_sizes = token_sizes.to(device=self.device)
        return pack_padded_sequence(
            sequence=sequence, token_sizes=token_sizes,
            batch_first=self.batch_first,
        )


class ReduceCattedSequences(PackProc):
    def __call__(self, sequences: List[CattedSequence], **kwargs) -> PackedSequence:
        return pack_catted_sequences(sequences=sequences, device=self.device)
