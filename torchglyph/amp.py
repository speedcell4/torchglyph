from typing import Type, Union

from torch import Tensor, autocast
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer


class amp(object):
    def __init__(self) -> None:
        super(amp, self).__init__()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def scale(self, loss: Tensor) -> Tensor:
        raise NotImplementedError

    def unscale(self, optimizer: Optimizer) -> None:
        raise NotImplementedError

    def step(self, optimizer: Optimizer) -> None:
        raise NotImplementedError


class fp32(amp):
    def scale(self, loss: Tensor) -> Tensor:
        return loss

    def unscale(self, optimizer: Optimizer) -> None:
        pass

    def step(self, optimizer: Optimizer) -> None:
        optimizer.step()
        optimizer.zero_grad()


class fp16(amp):
    def __init__(self) -> None:
        super(fp16, self).__init__()
        self.grad_scaler = GradScaler()

    def __enter__(self):
        self.env = autocast()
        self.env.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.env.__exit__(exc_type, exc_val, exc_tb)
        del self.env

    def scale(self, loss: Tensor) -> Tensor:
        return self.grad_scaler.scale(loss)

    def unscale(self, optimizer: Optimizer) -> None:
        return self.grad_scaler.unscale_(optimizer)

    def step(self, optimizer: Optimizer) -> None:
        self.grad_scaler.step(optimizer)
        self.grad_scaler.update()
        optimizer.zero_grad()


Amp = Union[
    Type[fp32],
    Type[fp16],
]
