import re
from typing import List

from torchglyph.proc.abc import Proc


class ToLower(Proc):
    def __call__(self, string: str, **kwargs) -> str:
        return string.lower()


class ToUpper(Proc):
    def __call__(self, string: str, **kwargs) -> str:
        return string.upper()


class ToCapitalize(Proc):
    def __call__(self, string: str, **kwargs) -> str:
        return string.capitalize()


class ReSub(Proc):
    def __init__(self, pattern: str, repl: str) -> None:
        super(ReSub, self).__init__()
        self.pattern = re.compile(pattern)
        self.repl = repl

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.pattern}',
            f'repl={self.repl}',
        ])

    def __call__(self, string: str, **kwargs) -> str:
        return re.sub(pattern=self.pattern, repl=self.repl, string=string)


class ReSplit(Proc):
    def __init__(self, pattern: str, max_count: int = 0) -> None:
        super(ReSplit, self).__init__()
        self.pattern = re.compile(pattern)
        self.max_count = max_count

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.pattern}',
            f'max_count={self.max_count}',
        ])

    def __call__(self, string: str, **kwargs) -> List[str]:
        return re.split(pattern=self.pattern, string=string, maxsplit=self.max_count)


class ReMatch(Proc):
    def __init__(self, pattern: str) -> None:
        super(ReMatch, self).__init__()
        self.pattern = re.compile(pattern)

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.pattern}',
        ])

    def __call__(self, string: str, **kwargs) -> List[str]:
        return re.findall(pattern=self.pattern, string=string)
