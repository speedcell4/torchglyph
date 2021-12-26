from logging import getLogger

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR as _CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR as _LambdaLR

__all__ = [
    'SchedulerMixin',
    'Fixed', 'InverseDecay',
]

logger = getLogger(__name__)


class SchedulerMixin(object):
    def __init__(self) -> None:
        self.batch = 1
        self.epoch = 1

    def batch_step(self) -> None:
        self.batch += 1

    def epoch_step(self) -> None:
        self.epoch += 1

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def report_lr(self) -> None:
        lr = ' | '.join(f'{lr:.6f}' for lr in self.get_lr())
        logger.info(f'epoch {self.epoch} | batch {self.batch} | learning rate => [{lr}]')


class InverseDecay(_LambdaLR, SchedulerMixin):
    def __init__(self, gamma: float = 0.05, *, optimizer: Optimizer, **_) -> None:
        SchedulerMixin.__init__(self)
        _LambdaLR.__init__(
            self, optimizer=optimizer,
            lr_lambda=lambda epoch: 1. / (1. + gamma * epoch),
        )
        self.gamma = gamma

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}'

    def epoch_step(self) -> None:
        super(InverseDecay, self).epoch_step()
        self.step()
        self.report_lr()


class Fixed(InverseDecay):
    def __init__(self, *, optimizer: Optimizer, **_) -> None:
        super(Fixed, self).__init__(gamma=0., optimizer=optimizer)


class CosineAnnealingLR(_CosineAnnealingLR, SchedulerMixin):
    def __init__(self, min_lr: float, *, optimizer: Optimizer, num_iterations: int) -> None:
        _CosineAnnealingLR.__init__(
            self, optimizer=optimizer,
            T_max=num_iterations, eta_min=min_lr,
        )
        SchedulerMixin.__init__(self)

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.T_max}',
            f'{self.eta_min}',
        ])

    def epoch_step(self) -> None:
        super(CosineAnnealingLR, self).epoch_step()
        self.step()
        self.report_lr()
