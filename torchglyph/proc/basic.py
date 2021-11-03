import re
from typing import Pattern, Any, List, Tuple, Set, Union, Sized

from torchglyph.proc.abc import Proc, Map

__all__ = [
    'ToStr', 'ToInt', 'ToBool', 'ToFloat',
    'ToSet', 'ToList', 'ToTuple', 'ToSize',
    'ToLower', 'ToUpper', 'ToCapitalized', 'RegexSub',
    'Prepend', 'Append',
]


class ToStr(Map):
    def map(self, data: Any, **kwargs) -> str:
        return str(data)


class ToInt(Map):
    def map(self, data: Any, **kwargs) -> int:
        return int(data)


class ToBool(Map):
    def map(self, data: Any, **kwargs) -> bool:
        return bool(data)


class ToFloat(Map):
    def map(self, data: Any, **kwargs) -> float:
        return float(data)


class ToSet(Proc):
    def __call__(self, data: Any, **kwargs) -> Set[Any]:
        return set(data)


class ToList(Proc):
    def __call__(self, data: Any, **kwargs) -> List[Any]:
        return list(data)


class ToTuple(Proc):
    def __call__(self, data: Any, **kwargs) -> Tuple[Any, ...]:
        return tuple(data)


class ToSize(Proc):
    def __call__(self, data: Sized, **kwargs) -> int:
        return len(data)


class ToLower(Map):
    def map(self, string: str, **kwargs) -> str:
        return string.lower()


class ToUpper(Map):
    def map(self, string: str, **kwargs) -> str:
        return string.upper()


class ToCapitalized(Map):
    def map(self, string: str, **kwargs) -> str:
        return string.capitalize()


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
