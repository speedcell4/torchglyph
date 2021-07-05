from typing import Optional, Tuple, Union

import torch
from torch import Tensor

Num = Union[int, bool, float]

DType = Optional[torch.dtype]
Device = Optional[torch.device]

CattedSequence = Tuple[Tensor, Tensor]
PaddedSequence = Tuple[Tensor, Tensor]
