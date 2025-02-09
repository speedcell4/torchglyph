from torch import Tensor, nn

from torchglyph.nn.criterion.utils import get_scaling


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, label_smoothing: float = 0.0, ignore_index: int = -100) -> None:
        super(CrossEntropyLoss, self).__init__(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction='mean',
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'ignore_index={self.ignore_index}',
            f'label_smoothing={self.label_smoothing}',
        ])

    def forward(self, tensor: Tensor, labels: Tensor) -> Tensor:
        loss = super(CrossEntropyLoss, self).forward(
            tensor.flatten(end_dim=-2),
            labels.flatten(end_dim=-1),
        )

        mask = labels != self.ignore_index
        scaling = get_scaling(mask.long().sum().detach().cpu().item())

        return loss * scaling
