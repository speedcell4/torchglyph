from logging import getLogger
from typing import Type, Union

from torch.optim import Optimizer, lr_scheduler

logger = getLogger(__name__)


class LambdaLR(lr_scheduler.LambdaLR):
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def step(self, epoch: int = None) -> None:
        super(LambdaLR, self).step(epoch=epoch)

    def report_lr(self) -> None:
        for group, lr in enumerate(self.get_last_lr()):
            logger.info(f'group {group} | lr => {lr:.10f}')


class ConstantScheduler(LambdaLR):
    def __init__(self, num_training_steps: int = 20_0000, num_warmup_steps: int = 5000, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))
            return 1.0

        super(ConstantScheduler, self).__init__(
            optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch,

        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_warmup_steps={self.num_warmup_steps}',
            f'num_training_steps={self.num_training_steps}',
        ])


class LinearScheduler(LambdaLR):
    def __init__(self, num_training_steps: int = 20_0000, num_warmup_steps: int = 5000, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))
            return max(0., (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps))

        super(LinearScheduler, self).__init__(
            optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch,
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_warmup_steps={self.num_warmup_steps}',
            f'num_training_steps={self.num_training_steps}',
        ])


class InverseSquareRootScheduler(LambdaLR):
    def __init__(self, num_training_steps: int = 20_0000, num_warmup_steps: int = 5000, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))
            return max(0., (num_warmup_steps / max(1, current_step)) ** 0.5)

        super(InverseSquareRootScheduler, self).__init__(
            optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch,
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_warmup_steps={self.num_warmup_steps}',
            f'num_training_steps={self.num_training_steps}',
        ])


Schedulers = Union[
    Type[ConstantScheduler],
    Type[LinearScheduler],
    Type[InverseSquareRootScheduler],
]
