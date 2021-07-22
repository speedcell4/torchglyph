from typing import Tuple, Union, Any, Set, List

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Number, _dtype as DType, Device

__all__ = [
    'Number', 'Tensor', 'Tensors', 'DType', 'Device',
    'CattedSequence', 'PackedSequence', 'PaddedSequence', 'Sequence',
]

Tensors = Union[Set[Any], List[Any], Tuple[Any, ...]]

PackedSequence = PackedSequence
CattedSequence = Tuple[Tensor, Tensor]
PaddedSequence = Union[Tensor, Tuple[Tensor, Tensor]]
Sequence = Union[PackedSequence, CattedSequence, PaddedSequence]
