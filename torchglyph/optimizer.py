from logging import getLogger
from typing import Set, Tuple, Type, Union

from torch import nn, optim

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


def group_params(modules: Tuple[nn.Module, ...], ignores: Tuple[Type[nn.Module], ...] = IGNORES):
    visited, require, without = set(), set(), set()

    def recur(module: nn.Module):
        if module not in visited:
            visited.add(module)

            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    if isinstance(module, ignores) or 'bias' in name:
                        without.add(param)
                    else:
                        require.add(param)

            for mod in module._modules.values():
                recur(module=mod)

    for m in modules:
        recur(module=m)

    return require, without


def log_params(*modules: nn.Module, require: Set[nn.Parameter], without: Set[nn.Parameter]):
    for module in modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                logger.critical(f'{name} {tuple(param.size())} -> no grad')
            elif param in require:
                logger.info(f'{name} {tuple(param.size())} -> decay')
            elif param in without:
                logger.info(f'{name} {tuple(param.size())} -> no decay')
            else:
                logger.error(f'{name} {tuple(param.size())} is not registered')


class SGD(optim.SGD):
    def __init__(self, lr: float = 1e-3, momentum: float = 0.9, dampening: float = 0.0,
                 weight_decay: float = 1e-4, nesterov: bool = False, *,
                 modules: Tuple[nn.Module, ...], **kwargs) -> None:
        require, without = group_params(modules)
        log_params(*modules, require=require, without=without)

        super(SGD, self).__init__(
            lr=lr, momentum=momentum, dampening=dampening, nesterov=nesterov,
            params=[
                {'params': list(require), 'weight_decay': weight_decay},
                {'params': list(without), 'weight_decay': 0.0},
            ],
        )


class Adam(optim.AdamW):
    def __init__(self, lr: float = 3e-4, beta1: float = 0.9, beta2: float = 0.98,
                 weight_decay: float = 1e-4, amsgrad: bool = False, *,
                 modules: Tuple[nn.Module, ...], **kwargs) -> None:
        require, without = group_params(modules)
        log_params(*modules, require=require, without=without)

        super(Adam, self).__init__(
            lr=lr, betas=(beta1, beta2), amsgrad=amsgrad,
            params=[
                {'params': list(require), 'weight_decay': weight_decay},
                {'params': list(without), 'weight_decay': 0.0},
            ],
        )


Optimizer = Union[
    Type[SGD],
    Type[Adam],
]
