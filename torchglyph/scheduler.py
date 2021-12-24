from logging import getLogger

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

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


class InverseDecay(LambdaLR, SchedulerMixin):
    def __init__(self, gamma: float = 0.05, *, optimizer: Optimizer, **_) -> None:
        SchedulerMixin.__init__(self)
        LambdaLR.__init__(
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
