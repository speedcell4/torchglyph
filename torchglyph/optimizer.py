from typing import Tuple, List

from torch import nn, optim
from torchlatent.crf import CrfDecoder

__all__ = [
    'ignores_default',
    'params_with_decay',
    'params_without_decay',

    'SGD', 'Adam',
]


ignores_default = (
    nn.LayerNorm, nn.GroupNorm, nn.LocalResponseNorm,

    nn.SyncBatchNorm,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LazyBatchNorm1d, nn.LazyBatchNorm2d, nn.LazyBatchNorm3d,

    nn.InstanceNorm2d, nn.InstanceNorm3d, nn.InstanceNorm3d,
    nn.LazyInstanceNorm2d, nn.LazyInstanceNorm3d, nn.LazyInstanceNorm3d,

    CrfDecoder,
)


def params_with_decay(module: nn.Module, ignores: Tuple[nn.Module, ...] = ignores_default) -> List[nn.Parameter]:
    return [
        param for mod in module.modules() if not isinstance(mod, ignores)
        for name, param in mod.named_parameters(recurse=False) if 'bias' not in name
    ]


def params_without_decay(module: nn.Module, ignores: Tuple[nn.Module, ...] = ignores_default) -> List[nn.Parameter]:
    return [
        param for mod in module.modules()
        for name, param in mod.named_parameters(recurse=False)
        if isinstance(mod, ignores) or 'bias' in name
    ]


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
