from typing import Any, Union, List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_sequence, pad_packed_sequence

from torchglyph.proc import Proc, Chain, stoi
from torchglyph.vocab import Vocab


class ToDevice(Proc):
    Batch = Union[Tensor, PackedSequence, Tuple[Union[Tensor, PackedSequence], ...]]

    def __init__(self, device: Union[int, torch.device]) -> None:
        super(ToDevice, self).__init__()
        if isinstance(device, int):
            if device < 0:
                device = torch.device(f'cpu')
            else:
                device = torch.device(f'cuda:{device}')
        self.device = device

    def extra_repr(self) -> str:
        return f'{self.device}'

    def __call__(self, batch: Batch, vocab: Vocab, **kwargs) -> Batch:
        if isinstance(batch, (PackedSequence, Tensor)):
            return batch.to(self.device)
        return type(batch)([self(e, vocab=vocab) for e in batch])


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def extra_repr(self) -> str:
        return f'{self.dtype}'

    def __call__(self, data: Any, **kwargs) -> Tensor:
        try:
            return torch.tensor(data, dtype=self.dtype, requires_grad=False)
        except ValueError as err:
            if err.args[0] == "too many dimensions 'str'":
                raise ValueError(f"'{data}' can not be converted to {Tensor.__name__}")
            raise err


class CatTensors(Proc):
    def __init__(self, dim: int) -> None:
        super(CatTensors, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, data: List[Tensor], **kwargs) -> Tensor:
        return torch.cat(data, dim=self.dim)


class StackTensors(Proc):
    def __init__(self, dim: int) -> None:
        super(StackTensors, self).__init__()
        self.dim = dim

    def extra_repr(self) -> str:
        return f'dim={self.dim}'

    def __call__(self, data: List[Tensor], **kwargs) -> Tensor:
        return torch.stack(data, dim=self.dim)


class FlattenList(Proc):
    def __call__(self, data: List[List[Tensor]], **kwargs) -> List[Tensor]:
        return [d for datum in data for d in datum]


class PadSeq(Proc):
    def __init__(self, pad_token: Union[int, str], batch_first: bool) -> None:
        super(PadSeq, self).__init__()
        self.pad_token = pad_token
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        return f', '.join([
            f"'{self.pad_token}'",
            'batch_first' if self.batch_first else 'batch_second',
        ])

    def __call__(self, data: List[Tensor], vocab: Vocab, **kwargs) -> Tensor:
        return pad_sequence(
            data, batch_first=self.batch_first,
            padding_value=stoi(token=self.pad_token, vocab=vocab),
        )


class PackSeq(Proc):
    def __init__(self, enforce_sorted: bool) -> None:
        super(PackSeq, self).__init__()
        self.enforce_sorted = enforce_sorted

    def extra_repr(self) -> str:
        return f'sorted={self.enforce_sorted}'

    def __call__(self, data: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(data, enforce_sorted=self.enforce_sorted)


class PackPtrSeq(Proc):
    def __init__(self, enforce_sorted: bool) -> None:
        super(PackPtrSeq, self).__init__()
        self.enforce_sorted = enforce_sorted

    def extra_repr(self) -> str:
        return f'enforce_sorted={self.enforce_sorted}'

    def __call__(self, data: List[Tensor], **kwargs) -> PackedSequence:
        pack = pack_sequence(data, enforce_sorted=self.enforce_sorted)
        index = pack._replace(data=torch.arange(pack.data.size(0), device=pack.data.device))
        index, _ = pad_packed_sequence(index, batch_first=True)

        return pack_sequence([
            index[i, datum]
            for i, datum in enumerate(data)
        ], enforce_sorted=self.enforce_sorted)


class PadBlock(Proc):
    def __init__(self, pad_token: Union[str, int], batch_first: bool) -> None:
        super(PadBlock, self).__init__()
        self.pad_token = pad_token
        self.batch_first = batch_first

    def extra_repr(self) -> str:
        return f', '.join([
            f"'{self.pad_token}'",
            f'batch_first={self.batch_first}',
        ])

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

        if not self.batch_first:
            tensor = tensor.transpose(0, 1)
        return tensor.detach()


class PackBlock(Chain):
    def __init__(self, enforce_sorted: bool) -> None:
        super(PackBlock, self).__init__([
            FlattenList(),
            PackSeq(enforce_sorted=enforce_sorted),
        ])
