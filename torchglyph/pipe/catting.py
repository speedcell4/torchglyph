from typing import List

import torch
from torch.types import Device, Number
from torchrua import CattedSequence

from torchglyph.pipe import Pipe
from torchglyph.proc import ToTensor, CatSequence


class CattedNumListPipe(Pipe):
    def __init__(self, device: Device, dtype: torch.dtype = torch.long) -> None:
        super(CattedNumListPipe, self).__init__(
            pre=None,
            post=ToTensor(dtype=dtype),
            vocab=None,
            batch=CatSequence(device=device),
        )

    def inv(self, sequence: CattedSequence) -> List[List[Number]]:
        data, token_sizes = sequence

        data = data.detach().cpu().tolist()
        token_sizes = token_sizes.detach().cpu().tolist()

        out, start, end = [], 0, 0
        for token_size in token_sizes:
            start, end = end, end + token_size
            out.append(data[start:end])

        return data
