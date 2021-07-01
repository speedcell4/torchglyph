from typing import Any, List

from torchglyph.proc.abc import Proc

__all__ = [
    'ToTokenSize', 'Prepend', 'Append',
]


class ToTokenSize(Proc):
    def __call__(self, x: List[Any], **kwargs) -> int:
        return len(x)


class Prepend(Proc):
    def __init__(self, token: Any, num_times: int = 1) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_times = num_times

    def extra_repr(self) -> str:
        return f'{self.token} x {self.num_times} + [...]'

    def __call__(self, x: List[Any], **kwargs) -> List[Any]:
        return [self.token for _ in range(self.num_times)] + x


class Append(Proc):
    def __init__(self, token: Any, num_times: int = 1) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_times = num_times

    def extra_repr(self) -> str:
        return f'[...] + {self.token} x {self.num_times}'

    def __call__(self, x: List[Any], **kwargs) -> List[Any]:
        return x + [self.token for _ in range(self.num_times)]
