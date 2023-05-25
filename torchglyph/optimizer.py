from logging import getLogger
from typing import Set, Tuple, Type, Union

from torch import nn, optim

logger = getLogger(__name__)

ignores_default = (
    nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,

    nn.SyncBatchNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,

    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.InstanceNorm3d,
    nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d, nn.LazyInstanceNorm3d,
)


def divide_groups(module: nn.Module, ignores: Tuple[nn.Module, ...] = None):
    if ignores is None:
        ignores = ignores_default

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

    recur(mod=module)
    validate_groups(module, with_decay=with_decay, without_decay=without_decay)

    return list(with_decay), list(without_decay)


def validate_groups(module: nn.Module, with_decay: Set[nn.Parameter], without_decay: Set[nn.Parameter]) -> None:
    mapping = {param: name for name, param in module.named_parameters() if param.requires_grad}

    union = with_decay | without_decay
    intersection = with_decay & without_decay

    if len(intersection) > 0:
        for param in intersection:
            logger.warning(f'{mapping[param]} is in both groups')

    if len(union) < len(mapping):
        for param, name in mapping.items():
            if param not in union:
                logger.warning(f'{name} is not in any group')


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, dampening: float = 0.0,
                 weight_decay: float = 1e-4, nesterov: bool = False, *, params, **kwargs) -> None:
        super(SGD, self).__init__(
            params=params, lr=lr,
            momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov,
        )


class Adam(optim.AdamW):
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.98,
                 weight_decay: float = 1e-4, amsgrad: bool = False, *, params, **kwargs) -> None:
        super(Adam, self).__init__(
            params=params, lr=lr, betas=(beta1, beta2),
            weight_decay=weight_decay, amsgrad=amsgrad,
        )


Optimizers = Union[
    Type[SGD],
    Type[Adam],
]
