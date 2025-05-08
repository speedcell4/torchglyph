from logging import getLogger

from torch import nn, optim
from typing import Type, Union

logger = getLogger(__name__)

IGNORES = (
    nn.Embedding, nn.EmbeddingBag,

    nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,

    nn.SyncBatchNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,

    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.InstanceNorm3d,
    nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d, nn.LazyInstanceNorm3d,
)


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, dampening: float = 0.0,
                 weight_decay: float = 0, nesterov: bool = False, *,
                 params, **kwargs) -> None:
        super(SGD, self).__init__(
            params=params,
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            dampening=dampening, nesterov=nesterov,
            **kwargs
        )


class Adam(optim.AdamW):
    def __init__(self, lr: float = 1e-6, beta1: float = 0.9, beta2: float = 0.999,
                 weight_decay: float = 0, amsgrad: bool = False, eps: float = 1e-8, *,
                 params, **kwargs) -> None:
        super(Adam, self).__init__(
            params=params,
            lr=lr, betas=(beta1, beta2), weight_decay=weight_decay,
            amsgrad=amsgrad, eps=eps,
            **kwargs
        )


Optimizers = Union[
    Type[SGD],
    Type[Adam],
]
