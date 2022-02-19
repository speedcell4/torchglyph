import math
from logging import getLogger

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = [
    'ConstantScheduler',
    'LinearScheduler',
    'InverseSquareRootScheduler',
]

logger = getLogger(__name__)


class ConstantScheduler(LambdaLR):
    def __init__(self, num_training_steps: int, warmup_ratio: float = 0.05, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        num_warmup_steps = int(math.ceil(warmup_ratio * num_training_steps))
        logger.info(f'num_warmup_steps => {num_warmup_steps}')

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))
            return 1.0

        super(ConstantScheduler, self).__init__(
            optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch,
        )

    def __repr__(self) -> str:
        return ', '.join([
            f'num_warmup_steps={self.num_warmup_steps}',
            f'num_training_steps={self.num_training_steps}',
        ])


class LinearScheduler(LambdaLR):
    def __init__(self, num_training_steps: int, warmup_ratio: float = 0.05, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        num_warmup_steps = int(math.ceil(warmup_ratio * num_training_steps))
        logger.info(f'num_warmup_steps => {num_warmup_steps}')

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))
            return max(0., (num_training_steps - current_step) / max(1, num_training_steps - num_warmup_steps))

        super(LinearScheduler, self).__init__(
            optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch,
        )

    def __repr__(self) -> str:
        return ', '.join([
            f'num_warmup_steps={self.num_warmup_steps}',
            f'num_training_steps={self.num_training_steps}',
        ])


class InverseSquareRootScheduler(LambdaLR):
    def __init__(self, num_training_steps: int, warmup_ratio: float = 0.05, *,
                 optimizer: Optimizer, last_epoch: int = -1, **kwargs) -> None:
        num_warmup_steps = int(math.ceil(warmup_ratio * num_training_steps))
        logger.info(f'num_warmup_steps => {num_warmup_steps}')

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps

        def lr_lambda(current_step: int) -> float:
            if current_step < num_warmup_steps:
                return float(current_step / max(1, num_warmup_steps))
            return max(0., (num_warmup_steps / current_step) ** 0.5)

        super(InverseSquareRootScheduler, self).__init__(
            optimizer=optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch,
        )

    def __repr__(self) -> str:
        return ', '.join([
            f'num_warmup_steps={self.num_warmup_steps}',
            f'num_training_steps={self.num_training_steps}',
        ])
