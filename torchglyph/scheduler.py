from logging import getLogger
from typing import Union, Type

from torch import optim

logger = getLogger(__name__)

__all__ = [
    'Schedulers', 'SchedulerMixin',
    'Fixed', 'InverseDecay',
]


class SchedulerMixin(object):
    batch: int
    epoch: int

    def report_learning_rate(self) -> None:
        lr = ', '.join([f'{lr:.6f}' for lr in self.get_lr()])
        logger.info(f'learning rate => [{lr}]')

    def batch_step(self) -> None:
        raise NotImplementedError

    def epoch_step(self) -> None:
        raise NotImplementedError


class Fixed(optim.lr_scheduler.LambdaLR, SchedulerMixin):
    def __init__(self, *, optimizer: optim.Optimizer) -> None:
        super(Fixed, self).__init__(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 1.,
        )
        self.batch = 1
        self.epoch = 1

    def batch_step(self) -> None:
        self.batch += 1

    def epoch_step(self) -> None:
        self.epoch += 1
        self.report_learning_rate()


class InverseDecay(optim.lr_scheduler.LambdaLR, SchedulerMixin):
    def __init__(self, gamma: float = 0.05, *, optimizer: optim.Optimizer) -> None:
        super(InverseDecay, self).__init__(
            optimizer=optimizer,
            lr_lambda=lambda epoch: 1. / (1. + gamma * epoch)
        )
        self.batch = 1
        self.epoch = 1

    def batch_step(self) -> None:
        self.batch += 1

    def epoch_step(self) -> None:
        self.step()
        self.epoch += 1
        self.report_learning_rate()


Schedulers = Union[
    Type[Fixed],
    Type[InverseDecay],
]
