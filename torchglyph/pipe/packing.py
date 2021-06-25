from typing import Optional, Tuple, List

import torch
from torch.nn.utils.rnn import PackedSequence
from torchrua.padding import pad_packed_sequence

from torchglyph.pipe import Pipe, THRESHOLD
from torchglyph.proc import ToTensor, UpdateCounter, BuildVocab, StatsVocab, Numbering
from torchglyph.proc.packing import PackList

__all__ = [
    'PackListNumPipe', 'PackListStrPipe'
]


class PackListNumPipe(Pipe):
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.long) -> None:
        super(PackListNumPipe, self).__init__(
            pre=None,
            vocab=None,
            post=ToTensor(dtype=dtype),
            batch=PackList(device=device),
        )


class PackListStrPipe(PackListNumPipe):
    def __init__(self, device: torch.device,
                 unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = (),
                 threshold: int = THRESHOLD,
                 dtype: torch.dtype = torch.long) -> None:
        super(PackListStrPipe, self).__init__(device=device, dtype=dtype)
        self.with_(
            pre=UpdateCounter(),
            vocab=[
                BuildVocab(unk_token=unk_token, pad_token=pad_token, special_tokens=special_tokens),
                StatsVocab(threshold=threshold),
            ],
            post=Numbering() + ...,
        )

    def inv(self, sequence: PackedSequence) -> List[List[str]]:
        data, token_sizes = pad_packed_sequence(sequence=sequence, batch_first=True)
        data = data.cpu().detach().tolist()
        token_sizes = token_sizes.cpu().detach().tolist()

        return [[
            self.vocab.itos[data[index1][index2]] for index2 in range(token_size)]
            for index1, token_size in enumerate(token_sizes)
        ]
