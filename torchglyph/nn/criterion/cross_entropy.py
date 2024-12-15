from typing import Literal

from torch import Tensor, distributed, nn

from torchglyph.utils import all_gather_object


class CrossEntropy(nn.CrossEntropyLoss):
    def __init__(self, label_smoothing: float = 0,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean',
                 scaling: bool = True, *, ignore_index: int = -100) -> None:
        super(CrossEntropy, self).__init__(
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
            reduction=reduction,
        )
        self.scaling = scaling

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = super(CrossEntropy, self).forward(
            input=input.flatten(end_dim=-2),
            target=target.flatten(end_dim=-1),
        )

        if not distributed.is_initialized() or not self.scaling:
            return loss

        weight = (target != self.ignore_index).to(dtype=loss.dtype)
        weight = weight.sum().detach().cpu().item()
        weights = all_gather_object(weight)

        return loss * (sum(weights) / (len(weights) * weight))
