from logging import getLogger

from torch import optim

__all__ = [
    'SchedulerMixin',
    'Fixed', 'InverseDecay',
]

logger = getLogger(__name__)


class SchedulerMixin(object):
    batch: int
    epoch: int

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def report_learning_rate(self) -> None:
        lr = ', '.join([f'{lr:.6f}' for lr in self.get_lr()])
        logger.info(f'learning rate => [{lr}]')

    def batch_step(self) -> None:
        self.batch += 1

    def epoch_step(self) -> None:
        self.epoch += 1


class Fixed(optim.lr_scheduler.LambdaLR, SchedulerMixin):
    def __init__(self, *, optimizer: optim.Optimizer, **_) -> None:
        super(Fixed, self).__init__(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 1.,
        )
        self.batch = 1
        self.epoch = 1

    def epoch_step(self) -> None:
        super(Fixed, self).epoch_step()
        self.report_learning_rate()


class InverseDecay(optim.lr_scheduler.LambdaLR, SchedulerMixin):
    def __init__(self, gamma: float = 0.05, *, optimizer: optim.Optimizer, **_) -> None:
        super(InverseDecay, self).__init__(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 1. / (1. + gamma * epoch)
        )
        self.batch = 1
        self.epoch = 1

        self.gamma = gamma

    def extra_repr(self) -> str:
        return f'gamma={self.gamma}'

    def epoch_step(self) -> None:
        self.step()
        super(InverseDecay, self).epoch_step()
        self.report_learning_rate()
