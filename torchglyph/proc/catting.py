from typing import List, Any

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import cat_sequence, cat_packed_sequence, cat_padded_sequence

from torchglyph.proc.abc import Proc
from torchglyph.annotations import Device, CattedSequence

__all__ = [
    'CatProc',
    'CatSequences', 'CatPackedSequence', 'CatPaddedSequence',
]


class CatProc(Proc):
    def __init__(self, device: Device = None) -> None:
        super(CatProc, self).__init__()
        self.device = device

    def extra_repr(self) -> str:
        if self.device is not None:
            return f'device={self.device}'
        return ''

    def __call__(self, sequence: Any, **kwargs) -> CattedSequence:
        raise NotImplementedError


class CatSequences(CatProc):
    def __call__(self, sequences: List[Tensor], **kwargs) -> CattedSequence:
        return cat_sequence(sequences=sequences, device=self.device)


class CatPackedSequence(CatProc):
    def __call__(self, sequence: PackedSequence, **kwargs) -> CattedSequence:
        return cat_packed_sequence(sequence=sequence, device=self.device)


class CatPaddedSequence(CatProc):
    def __init__(self, batch_first: bool = True, device: Device = None) -> None:
        super(CatPaddedSequence, self).__init__(device=device)
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        args = [
            f'batch_first={self.batch_first}'
        ]
        super_args = super(CatPaddedSequence, self).extra_repr()
        if super_args != '':
            args.append(super_args)
        return ', '.join(args)

    def __call__(self, sequence: PackedSequence, **kwargs) -> CattedSequence:
        sequence, token_sizes = sequence
        return cat_padded_sequence(
            sequence=sequence, token_sizes=token_sizes,
            batch_first=self.batch_first, device=self.device,
        )
