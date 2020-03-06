from typing import Any, Union, List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_sequence

from torchglyph.proc.abc import Proc
from torchglyph.proc.vocab import Numbering
from torchglyph.proc.utiles import stoi
from torchglyph.vocab import Vocab


class ToDevice(Proc):
    Batch = Union[PackedSequence, Tensor, Tuple[Tensor, ...]]

    def __init__(self, device: Union[int, torch.device]) -> None:
        super(ToDevice, self).__init__()
        if isinstance(device, int):
            if device < 0:
                device = torch.device(f'cpu')
            else:
                device = torch.device(f'cuda:{device}')
        self.device = device

    def __call__(self, batch: Batch, vocab: Vocab, **kwargs) -> Batch:
        if isinstance(batch, (PackedSequence, Tensor)):
            return batch.to(self.device)
        return type(batch)([self(e, vocab=vocab) for e in batch])


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype = torch.long) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def __call__(self, ins: Any, **kwargs) -> Tensor:
        try:
            return torch.tensor(ins, dtype=self.dtype, requires_grad=False)
        except ValueError as err:
            if err.args[0] == "too many dimensions 'str'":
                raise ValueError(f'did you forget {Numbering.__name__} before {ToTensor.__name__}?')
            raise err


class PadSeq(Proc):
    def __init__(self, pad_token: Union[int, str], batch_first: bool = True) -> None:
        super(PadSeq, self).__init__()
        self.pad_token = pad_token
        self.batch_first = batch_first

    def __call__(self, batch: List[Tensor], vocab: Vocab, **kwargs) -> Tensor:
        return pad_sequence(
            batch, batch_first=self.batch_first,
            padding_value=stoi(token=self.pad_token, vocab=vocab),
        )


class PackSeq(Proc):
    def __init__(self, enforce_sorted: bool = False) -> None:
        super(PackSeq, self).__init__()
        self.enforce_sorted = enforce_sorted

    def __call__(self, data: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(data, enforce_sorted=self.enforce_sorted)


class PadSub(Proc):
    def __init__(self, pad_token: str, batch_first: bool = True) -> None:
        super(PadSub, self).__init__()
        self.pad_token = pad_token
        self.batch_first = batch_first

    def __call__(self, data: List[Tensor], vocab: Vocab, **kwargs) -> Tensor:
        dim1, dim2 = zip(*[d.size() for d in data])
        dim0 = len(data)
        dim1 = max(dim1)
        dim2 = max(dim2)

        tensor = torch.full(
            (dim0, dim1, dim2), dtype=torch.long,
            fill_value=stoi(token=self.pad_token, vocab=vocab),
        )
        for index, d in enumerate(data):
            dim1, dim2 = d.size()
            tensor[index, :dim1, :dim2] = d

        return tensor.detach()


class PackSub(Proc):
    def __init__(self, enforce_sorted: bool = False) -> None:
        super(PackSub, self).__init__()
        self.enforce_sorted = enforce_sorted

    def __call__(self, data: List[List[Tensor]], **kwargs) -> PackedSequence:
        char = [torch.cat(words, dim=0) for words in data]
        return pack_sequence(char, enforce_sorted=self.enforce_sorted)
