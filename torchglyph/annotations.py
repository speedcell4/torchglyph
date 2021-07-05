from typing import Optional, Tuple, Union, Any, Set, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

Num = Union[int, bool, float]
Container = Union[Set[Any], List[Any], Tuple[Any, ...]]

DType = Optional[torch.dtype]
Device = Optional[torch.device]

PackedSequence = PackedSequence
CattedSequence = Tuple[Tensor, Tensor]
PaddedSequence = Union[Tensor, Tuple[Tensor, Tensor]]
Sequence = Union[PackedSequence, CattedSequence, PaddedSequence]
