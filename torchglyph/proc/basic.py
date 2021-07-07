import re
from typing import Pattern, Any, List

from torchglyph.annotations import Container
from torchglyph.proc import Proc
from torchglyph.proc.abc import Map

__all__ = [
    'ToInt', 'ToBool', 'ToFloat',
    'ToLower', 'ToUpper', 'ToCapitalized', 'RegexSub',
    'ToList', 'ToLen', 'Prepend', 'Append',
]


class ToInt(Map):
    def map(self, token: str, **kwargs) -> int:
        return int(token)


class ToBool(Map):
    def map(self, token: str, **kwargs) -> bool:
        return bool(token)


class ToFloat(Map):
    def map(self, token: str, **kwargs) -> float:
        return float(token)


class ToLower(Map):
    def map(self, token: str, **kwargs) -> str:
        return token.lower()


class ToUpper(Map):
    def map(self, token: str, **kwargs) -> str:
        return token.upper()


class ToCapitalized(Map):
    def map(self, token: str, **kwargs) -> str:
        return token.capitalize()


class RegexSub(Map):
    def __init__(self, pattern: Pattern, repl: str) -> None:
        super(RegexSub, self).__init__()
        self.pattern = pattern
        self.repl = repl

    def extra_repr(self) -> str:
        return f'{self.pattern} -> {self.repl}'

    def map(self, token: str, **kwargs) -> str:
        return re.sub(pattern=self.pattern, repl=self.repl, string=token)


class ToList(Proc):
    def __call__(self, data: Any, **kwargs) -> List[Any]:
        return list(data)


class ToLen(Proc):
    def __call__(self, data: Container, **kwargs) -> int:
        return len(data)


class Prepend(Proc):
    def __init__(self, token: Any, num_times: int = 1) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_times = num_times

    def extra_repr(self) -> str:
        return f'{self.token} x {self.num_times} + [...]'

    def __call__(self, data: Container, **kwargs) -> Container:
        return type(data)([self.token for _ in range(self.num_times)] + data)


class Append(Proc):
    def __init__(self, token: Any, num_times: int = 1) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_times = num_times

    def extra_repr(self) -> str:
        return f'[...] + {self.token} x {self.num_times}'

    def __call__(self, data: Container, **kwargs) -> Container:
        return type(data)(data + [self.token for _ in range(self.num_times)])
