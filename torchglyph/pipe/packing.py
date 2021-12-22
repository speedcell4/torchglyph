from typing import Optional, Tuple, List

import torch
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number
from torchrua.padding import pad_packed_sequence

from torchglyph.pipe.abc import Pipe
from torchglyph.proc.abc import Lift
from torchglyph.proc.basic import ToList
from torchglyph.proc.catting import CatSequence
from torchglyph.proc.collating import ToTensor
from torchglyph.proc.packing import PackSequence, ReduceCattedSequences
from torchglyph.proc.vocab import UpdateCounter, BuildVocab, StatsVocab, Numbering

__all__ = [
    'PackListNumPipe', 'PackListListNumPipe',
    'PackListStrPipe', 'PackListListStrPipe',
]


class PackListNumPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(PackListNumPipe, self).__init__()
        self.with_(
            post=ToTensor(dtype=dtype),
            batch=PackSequence(device=device),
        )

    def inv(self, sequence: PackedSequence) -> List[List[Number]]:
        data, token_sizes = pad_packed_sequence(sequence=sequence, batch_first=True)

        data = data.detach().cpu().tolist()
        token_sizes = token_sizes.detach().cpu().tolist()

        return [
            [data[index1][index2] for index2 in range(token_size)]
            for index1, token_size in enumerate(token_sizes)
        ]


class PackListStrPipe(PackListNumPipe):
    def __init__(self, device: Device,
                 unk_token: Optional[str], special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 10, dtype: torch.dtype = torch.long) -> None:
        super(PackListStrPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )

    def inv(self, sequence: PackedSequence) -> List[List[str]]:
        assert sequence.data.dim() == 1, f'{sequence.data.dim()} != 1'

        return [
            [self.vocab.inv(index) for index in indices]
            for indices in super(PackListStrPipe, self).inv(sequence)
        ]


class PackListListNumPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(PackListListNumPipe, self).__init__(
            pre=None,
            vocab=None,
            post=Lift(ToTensor(dtype=dtype)) + CatSequence(device=None),
            batch=ReduceCattedSequences(device=device),
        )


class PackListListStrPipe(PackListListNumPipe):
    def __init__(self, device: Device,
                 unk_token: Optional[str], special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = 10, dtype: torch.dtype = torch.long) -> None:
        super(PackListListStrPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=Lift(ToList() + UpdateCounter()),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=None, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Lift(Numbering()) + ...,
        )
