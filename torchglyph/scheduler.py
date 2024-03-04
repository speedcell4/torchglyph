from logging import getLogger
from typing import Type, Union

from torch.optim import Optimizer, lr_scheduler

from torchglyph import DEBUG

logger = getLogger(__name__)


class LambdaLR(lr_scheduler.LambdaLR):
    def __init__(self, num_training_steps: int = 20 if DEBUG else 5_0000,
                 num_warmup_steps: int = 20 if DEBUG else 4000, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

        super(LambdaLR, self).__init__(
            optimizer=optimizer, lr_lambda=self.lr_lambda,
            last_epoch=last_epoch,
        )

    def lr_lambda(self, step: int) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_training_steps={self.num_training_steps}',
            f'num_warmup_steps={self.num_warmup_steps}',
        ])


class ConstantScheduler(LambdaLR):
    def lr_lambda(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return float(step / max(1.0, self.num_warmup_steps))

        return 1.0


class LinearScheduler(LambdaLR):
    def lr_lambda(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return float(step / max(1.0, self.num_warmup_steps))

        return max(0., (self.num_training_steps - step) / max(1.0, self.num_training_steps - self.num_warmup_steps))


class InverseSquareRootScheduler(LambdaLR):
    def lr_lambda(self, step: int) -> float:
        if step < self.num_warmup_steps:
            return float(step / max(1.0, self.num_warmup_steps))

        return max(0., (self.num_warmup_steps / max(1.0, step)) ** 0.5)


Scheduler = Union[
    Type[ConstantScheduler],
    Type[LinearScheduler],
    Type[InverseSquareRootScheduler],
]
