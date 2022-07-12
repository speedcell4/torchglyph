import re

from torchglyph.proc.abc import Proc


class ToLower(Proc):
    def __call__(self, data: str, **kwargs) -> str:
        return data.lower()


class ToUpper(Proc):
    def __call__(self, data: str, **kwargs) -> str:
        return data.upper()


class ToCapitalize(Proc):
    def __call__(self, data: str, **kwargs) -> str:
        return data.capitalize()


class RegexSub(Proc):
    def __init__(self, pattern: str, repl: str) -> None:
        super(RegexSub, self).__init__()
        self.pattern = re.compile(pattern)
        self.repl = repl

    def extra_repr(self) -> str:
        return f'{self.pattern} -> {self.repl}'

    def __call__(self, data: str, **kwargs) -> str:
        return re.sub(pattern=self.pattern, repl=self.repl, string=data)
