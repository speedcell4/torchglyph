import re
from typing import Pattern

from torchglyph.proc.abc import Flatten


class ToUpper(Flatten):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.upper()


class ToLower(Flatten):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.lower()


class ToCapitalized(Flatten):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.capitalize()


class RegexSubs(Flatten):
    def __init__(self, pattern: Pattern, repl_token: str) -> None:
        super(RegexSubs, self).__init__()
        self.pattern = pattern
        self.repl_token = repl_token

    def process(self, data: str, *args, **kwargs) -> str:
        return re.sub(self.pattern, self.repl_token, data)


class ReplaceDigits(RegexSubs):
    def __init__(self, repl_token: str) -> None:
        super(ReplaceDigits, self).__init__(
            pattern=re.compile(r'\d+'),
            repl_token=repl_token,
        )
