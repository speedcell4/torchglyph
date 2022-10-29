from typing import Literal

from torch import Tensor
from torch import nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 label_smoothing: float = 0.0, *, ignore_index: int = -100) -> None:
        super(CrossEntropyLoss, self).__init__(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super(CrossEntropyLoss, self).forward(
            input=input.view((-1, input.size()[-1])),
            target=target.view(-1),
        )
