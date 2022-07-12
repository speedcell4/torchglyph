import re
from typing import Pattern, Any, List, Tuple, Set, Union, Sized

from torchglyph.proc.abc import Proc, Map


class ToSize(Proc):
    def __call__(self, data: Sized, **kwargs) -> int:
        return len(data)


class RegexSub(Map):
    def __init__(self, pattern: Pattern, repl: str) -> None:
        super(RegexSub, self).__init__()
        self.pattern = pattern
        self.repl = repl

    def extra_repr(self) -> str:
        return f'{self.pattern} -> {self.repl}'

    def map(self, string: str, **kwargs) -> str:
        return re.sub(pattern=self.pattern, repl=self.repl, string=string)


class Prepend(Proc):
    Container = Union[Set[Any], List[Any], Tuple[Any, ...]]

    def __init__(self, token: Any, num_times: int = 1) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_times = num_times

    def extra_repr(self) -> str:
        return f'{self.token}(x{self.num_times}) + ...'

    def __call__(self, sequence: Container, **kwargs) -> Container:
        return type(sequence)([self.token for _ in range(self.num_times)] + sequence)


class Append(Proc):
    Container = Union[Set[Any], List[Any], Tuple[Any, ...]]

    def __init__(self, token: Any, num_times: int = 1) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_times = num_times

    def extra_repr(self) -> str:
        return f'... + {self.token}(x{self.num_times})'

    def __call__(self, sequence: Container, **kwargs) -> Container:
        return type(sequence)(sequence + [self.token for _ in range(self.num_times)])
