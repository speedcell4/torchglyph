from typing import Any, Union, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence, PackedSequence, pack_sequence, pad_packed_sequence

from torchglyph.proc import Proc, Chain, stoi
from torchglyph.vocab import Vocab


class ToDevice(Proc):
    Data = Union[int, float, bool, Tensor, PackedSequence]
    Batch = Union[Data, Tuple[Data, ...]]

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

    def __call__(self, batch: Batch, *, vocab: Vocab, **kwargs) -> Batch:
        if isinstance(batch, (Tensor, PackedSequence)):
            return batch.to(device=self.device)
        if isinstance(batch, (list, tuple, set)):
            return type(batch)([self(example, vocab=vocab, **kwargs) for example in batch])
        return batch


class ToTensor(Proc):
    def __init__(self, dtype: torch.dtype) -> None:
        super(ToTensor, self).__init__()
        self.dtype = dtype

    def extra_repr(self) -> str:
        return f'{self.dtype}'

    def __call__(self, data: Any, **kwargs) -> Tensor:
        try:
            if torch.is_tensor(data):
                tensor = data.clone()
            elif isinstance(data, np.ndarray):
                tensor = torch.from_numpy(data).clone()
            else:
                tensor = torch.tensor(data)
            return tensor.to(dtype=self.dtype).requires_grad_(False)
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
