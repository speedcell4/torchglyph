from typing import Iterable, Union, Type

from torch import nn, optim


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9,
                 dampening: float = 0., weight_decay: float = 1e-6,
                 nesterov: bool = False, *, params: Iterable[nn.Parameter]) -> None:
        super(SGD, self).__init__(
            params=params, lr=lr,
            momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
        )


Optimizers = Union[
    Type[SGD],
]
