from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from torchglyph.proc.abc import Proc


class BuildContiguousSub(Proc):
    def __init__(self, seq_token: str) -> None:
        super(BuildContiguousSub, self).__init__()
        self.seq_token = seq_token

    def extra_repr(self) -> str:
        return repr(self.seq_token)

    def __call__(self, tokens: List[str], **kwargs) -> List[str]:
        zs = []
        for token in tokens:
            zs.extend(list(token))
            zs.append(self.seq_token)
        return zs[:-1]


class BuildContiguousSubPtr(Proc):
    def __call__(self, lengths: List[int], **kwargs) -> Tuple[List[int], List[int]]:
        indices = [0]
        for length in lengths:
            indices.append(indices[-1] + length + 1)
        return [index - 2 for index in indices[1:]], indices[:-1]


class PackContiguousSubPtr(Proc):
    def __init__(self, enforce_sorted: bool) -> None:
        super(PackContiguousSubPtr, self).__init__()
        self.enforce_sorted = enforce_sorted

    def extra_repr(self) -> str:
        return f'enforce_sorted={self.enforce_sorted}'

    def __call__(self, indices: List[Tuple[Tensor, Tensor]], **kwargs) -> Tuple[PackedSequence, PackedSequence]:
        fidx, bidx = zip(*indices)

        pack = pack_sequence([
            torch.empty((f.max().item() + 1,), dtype=torch.long) for f in fidx
        ], enforce_sorted=self.enforce_sorted)
        indices = pack._replace(data=torch.arange(pack.data.size(0), device=pack.data.device))
        indices, _ = pad_packed_sequence(indices, batch_first=True)

        fidx = pack_sequence([
            indices[i, f] for i, f in enumerate(fidx)
        ], enforce_sorted=self.enforce_sorted)
        bidx = pack_sequence([
            indices[i, b] for i, b in enumerate(bidx)
        ], enforce_sorted=self.enforce_sorted)
        return fidx, bidx
