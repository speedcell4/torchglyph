from logging import getLogger
from typing import Set, Tuple, Type, Union

from torch import nn, optim

logger = getLogger(__name__)

IGNORES = (
    nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,

    nn.SyncBatchNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,

    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.InstanceNorm3d,
    nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d, nn.LazyInstanceNorm3d,
)


def group_parameters(*modules: nn.Module, ignores: Tuple[Type[nn.Module], ...] = IGNORES):
    memory = set()
    with_decay = set()
    without_decay = set()

    def recur(mod: nn.Module):
        if mod in memory:
            return

        memory.add(mod)

        for name, param in mod.named_parameters(recurse=False):
            if param.requires_grad:
                if isinstance(mod, ignores) or 'bias' in name:
                    without_decay.add(param)
                else:
                    with_decay.add(param)

        for m in mod._modules.values():
            recur(mod=m)

    for module in modules:
        recur(mod=module)

    return with_decay, without_decay


def log_parameters(module: nn.Module, with_decay: Set[nn.Parameter], without_decay: Set[nn.Parameter]):
    for name, param in module.named_parameters():
        if not param.requires_grad:
            logger.critical(f'{name} requires no grad')
        elif param in with_decay:
            logger.info(f'{name} with decay')
        elif param in without_decay:
            logger.info(f'{name} without decay')
        else:
            logger.error(f'{name} is not registered')


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, dampening: float = 0.0,
                 weight_decay: float = 1e-4, nesterov: bool = False, *modules: nn.Module, **kwargs) -> None:
        with_decay, without_decay = group_parameters(*modules)
        for module in modules:
            log_parameters(module, with_decay=with_decay, without_decay=without_decay)

        super(SGD, self).__init__(
            lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
            params=[
                {'params': list(with_decay), 'weight_decay': weight_decay},
                {'params': list(without_decay), 'weight_decay': 0.0},
            ],
        )


class Adam(optim.AdamW):
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.98,
                 weight_decay: float = 1e-4, amsgrad: bool = False, *modules: nn.Module, **kwargs) -> None:
        with_decay, without_decay = group_parameters(*modules)
        for module in modules:
            log_parameters(module, with_decay=with_decay, without_decay=without_decay)

        super(Adam, self).__init__(
            lr=lr, betas=(beta1, beta2), amsgrad=amsgrad,
            params=[
                {'params': list(with_decay), 'weight_decay': weight_decay},
                {'params': list(without_decay), 'weight_decay': 0.0},
            ],
        )


Optimizer = Union[
    Type[SGD],
    Type[Adam],
]
