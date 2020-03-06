from typing import List, Union, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_sequence

from torchglyph.proc import BatchProc
from torchglyph.vocab import Vocab


class PadTokBatch(BatchProc):
    Batch = List[int]

    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(PadTokBatch, self).__init__()
        self.dtype = dtype

    def __call__(self, batch: Batch, vocab: Vocab) -> Tensor:
        return torch.tensor(batch, dtype=self.dtype, requires_grad=False)


class PadSeqBatch(BatchProc):
    Batch = List[Tensor]

    def __init__(self, pad_token: Union[int, str], batch_first: bool = True) -> None:
        super(PadSeqBatch, self).__init__()
        self.pad_token = pad_token
        self.batch_first = batch_first

    def __call__(self, batch: Batch, vocab: Vocab) -> Tensor:
        if isinstance(self.pad_token, str):
            assert vocab is not None, 'Vocab is not built yet'
            assert self.pad_token in vocab.stoi, f'{self.pad_token} is not in Vocab'
            pad_idx = vocab.stoi[self.pad_token]
        else:
            pad_idx = self.pad_token

        return pad_sequence(
            batch, batch_first=self.batch_first,
            padding_value=pad_idx,
        )


class PackSeqBatch(BatchProc):
    Batch = List[Tensor]

    def __init__(self, enforce_sorted: bool = False) -> None:
        super(PackSeqBatch, self).__init__()
        self.enforce_sorted = enforce_sorted

    def __call__(self, batch: Batch, vocab: Vocab) -> PackedSequence:
        return pack_sequence(batch, enforce_sorted=self.enforce_sorted)


class PadSubBatch(BatchProc):
    Batch = List[Tensor]

    def __init__(self, pad_token: str, batch_first: bool = True) -> None:
        super(PadSubBatch, self).__init__()
        self.pad_token = pad_token
        self.batch_first = batch_first

    def __call__(self, batch: Batch, vocab: Vocab) -> Tensor:
        if isinstance(self.pad_token, str):
            assert vocab is not None
            assert self.pad_token in vocab.stoi

            pad_idx = vocab.stoi[self.pad_token]
        else:
            pad_idx = self.pad_token

        dim1, dim2 = zip(*[d.size() for d in batch])
        dim0 = len(batch)
        dim1 = max(dim1)
        dim2 = max(dim2)

        tensor = torch.full((dim0, dim1, dim2), fill_value=pad_idx, dtype=torch.long)
        for index, d in enumerate(batch):
            dim1, dim2 = d.size()
            tensor[index, :dim1, :dim2] = d

        return tensor.clone()


class PackSubBatch(BatchProc):
    Batch = List[List[Tensor]]

    def __init__(self, enforce_sorted: bool = False) -> None:
        super(PackSubBatch, self).__init__()
        self.enforce_sorted = enforce_sorted

    def __call__(self, batch: Batch, vocab: Vocab) -> PackedSequence:
        char = [torch.cat(words, dim=0) for words in batch]
        return pack_sequence(char, enforce_sorted=self.enforce_sorted)


class ToDevice(BatchProc):
    Batch = Union[PackedSequence, Tensor, Tuple[Tensor, ...]]

    def __init__(self, device: Union[int, torch.device]) -> None:
        super(ToDevice, self).__init__()
        if isinstance(device, int):
            if device < 0:
                device = torch.device(f'cpu')
            else:
                device = torch.device(f'cuda:{device}')
        self.device = device

    def __call__(self, batch: Batch, vocab: Vocab) -> Batch:
        if isinstance(batch, (PackedSequence, Tensor)):
            return batch.to(self.device)
        return type(batch)([self(e, vocab=vocab) for e in batch])
