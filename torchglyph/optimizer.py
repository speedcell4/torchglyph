from typing import Iterable

from torch import nn, optim

__all__ = [
    'SGD', 'Adam',
]


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, dampening: float = 0.0,
                 weight_decay: float = 1e-8, nesterov: bool = False, *,
                 params: Iterable[nn.Parameter], **_) -> None:
        super(SGD, self).__init__(
            params=params, lr=lr,
            momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
        )


class Adam(optim.AdamW):
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.98,
                 weight_decay: float = 1e-8, amsgrad: bool = False, *,
                 params: Iterable[nn.Parameter], **_) -> None:
        super(Adam, self).__init__(
            params=params, lr=lr, betas=(beta1, beta2),
            weight_decay=weight_decay, amsgrad=amsgrad,
        )
