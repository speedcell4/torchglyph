from typing import Tuple, List

import torch
from torch.nn.utils.rnn import PackedSequence
from torch.types import Device, Number
from torchrua.padding import pad_packed_sequence

from torchglyph.pipe.abc import Pipe
from torchglyph.proc.abc import Lift
from torchglyph.proc.cast import ToList
from torchglyph.proc.catting import CatSequence
from torchglyph.proc.packing import PackSequence, ComposeCattedSequences
from torchglyph.proc.tensor import ToTensor
from torchglyph.proc.vocab import CountTokenSequence, BuildVocab, StatsVocab, ToIndexSequence


class PackedNumListPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(PackedNumListPipe, self).__init__()
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


class PackedStrListPipe(PackedNumListPipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long,
                 unk_token: str = None, pad_token: str = None,
                 bos_token: str = None, eos_token: str = None,
                 special_tokens: Tuple[str, ...] = (), threshold: int = None) -> None:
        super(PackedStrListPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=CountTokenSequence(),
            vocab=[
                BuildVocab(
                    unk_token=unk_token, pad_token=pad_token,
                    bos_token=bos_token, eos_token=eos_token,
                    special_tokens=special_tokens,
                ),
                StatsVocab(n=threshold),
            ],
            post=ToIndexSequence() + ...,
        )

    def inv(self, sequence: PackedSequence) -> List[List[str]]:
        assert sequence.data.dim() == 1, f'{sequence.data.dim()} != 1'

        return self.vocab.inv(super(PackedStrListPipe, self).inv(sequence=sequence))


class PackedNumListListPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(PackedNumListListPipe, self).__init__(
            pre=None,
            vocab=None,
            post=Lift(ToTensor(dtype=dtype)) + CatSequence(device=None),
            batch=ComposeCattedSequences(device=device),
        )


class PackedStrListListPipe(PackedNumListListPipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long,
                 unk_token: str = None, pad_token: str = None,
                 bos_token: str = None, eos_token: str = None,
                 special_tokens: Tuple[str, ...] = (), threshold: int = None) -> None:
        super(PackedStrListListPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=Lift(ToList() + CountTokenSequence()),
            vocab=[
                BuildVocab(
                    unk_token=unk_token, pad_token=pad_token,
                    bos_token=bos_token, eos_token=eos_token,
                    special_tokens=special_tokens,
                ),
                StatsVocab(n=threshold),
            ],
            post=Lift(ToIndexSequence() + ToTensor(dtype=dtype)) + CatSequence(device=None),
            batch=ComposeCattedSequences(device=device)
        )
