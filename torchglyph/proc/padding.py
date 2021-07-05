from typing import List, Any

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import pad_sequence, pad_packed_sequence, pad_catted_sequence

from torchglyph.proc.abc import Proc
from torchglyph.proc.annotations import Device, Num, CattedSequence

__all__ = [
    'PadProc', 'PadSequence',
    'PadCattedSequence', 'PadPackedSequence',
]


class PadProc(Proc):
    def __init__(self, batch_first: bool = True, padding_value: Num = 0, device: Device = None) -> None:
        super(PadProc, self).__init__()
        self.batch_first = batch_first
        self.padding_value = padding_value
        self.device = device

    def extra_repr(self) -> str:
        args = [
            f'batch_first={self.batch_first}',
            f'padding_value={self.padding_value}',
        ]
        if self.device is not None:
            args.append(f'device={self.device}')
        return ', '.join(args)

    def __call__(self, sequence: Any, **kwargs) -> Tensor:
        raise NotImplementedError


class PadSequence(PadProc):
    def __call__(self, sequences: List[Tensor], **kwargs) -> Tensor:
        return pad_sequence(
            sequences=sequences, batch_first=self.batch_first,
            padding_value=self.padding_value, device=self.device,
        )


class PadPackedSequence(PadProc):
    def __call__(self, sequence: PackedSequence, **kwargs) -> Tensor:
        data, _ = pad_packed_sequence(
            sequence=sequence,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
        )
        return data


class PadCattedSequence(PadProc):
    def __call__(self, sequence: CattedSequence, **kwargs) -> Tensor:
        sequence, token_sizes = sequence
        return pad_catted_sequence(
            sequence=sequence, token_sizes=token_sizes,
            batch_first=self.batch_first,
            padding_value=self.padding_value,
        )
