from typing import Type, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import optimization


class Scheduler(LambdaLR):
    def __init__(self, num_training_steps: int = 200000,
                 num_warmup_steps: int = 4000, *, optimizer: Optimizer):
        super(Scheduler, self).__init__(
            optimizer=optimizer,
            lr_lambda=self.get_lr_lambda(
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
            )
        )

        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_training_steps={self.num_training_steps}',
            f'num_warmup_steps={self.num_warmup_steps}',
        ])

    def get_lr_lambda(self, **kwargs):
        raise NotImplementedError()


class ConstantScheduler(Scheduler):
    def get_lr_lambda(self, num_warmup_steps: int, **kwargs):
        def lr_lambda(current_step: int):
            return optimization._get_constant_schedule_with_warmup_lr_lambda(
                current_step=current_step,
                num_warmup_steps=num_warmup_steps,
            )

        return lr_lambda


class InverseSqrtScheduler(Scheduler):
    def get_lr_lambda(self, num_warmup_steps: int, **kwargs):
        def lr_lambda(current_step: int):
            return optimization._get_inverse_sqrt_schedule_lr_lambda(
                current_step=current_step,
                num_warmup_steps=num_warmup_steps,
                timescale=num_warmup_steps,
            )

        return lr_lambda


class CosineScheduler(Scheduler):
    def get_lr_lambda(self, num_warmup_steps: int, num_training_steps: int, **kwargs):
        def lr_lambda(current_step: int):
            return optimization._get_cosine_schedule_with_warmup_lr_lambda(
                current_step=current_step,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
                num_cycles=0.5,
                min_lr_rate=0.0,
            )

        return lr_lambda


class LinearScheduler(Scheduler):
    def get_lr_lambda(self, num_warmup_steps: int, num_training_steps: int, **kwargs):
        def lr_lambda(current_step: int):
            return optimization._get_linear_schedule_with_warmup_lr_lambda(
                current_step=current_step,
                num_training_steps=num_training_steps,
                num_warmup_steps=num_warmup_steps,
            )

        return lr_lambda


Schedulers = Union[
    Type[ConstantScheduler],
    Type[InverseSqrtScheduler],
    Type[CosineScheduler],
    Type[LinearScheduler],
]
