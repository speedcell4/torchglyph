import re
from typing import Pattern

from torchglyph.proc.abc import Recur


class ToUpper(Recur):
    def process(self, datum: str, *args, **kwargs) -> str:
        return datum.upper()


class ToLower(Recur):
    def process(self, datum: str, *args, **kwargs) -> str:
        return datum.lower()


class ToCapitalized(Recur):
    def process(self, datum: str, *args, **kwargs) -> str:
        return datum.capitalize()


class RegexSubs(Recur):
    def __init__(self, pattern: Pattern, repl_token: str) -> None:
        super(RegexSubs, self).__init__()
        self.pattern = pattern
        self.repl_token = repl_token

    def extra_repr(self) -> str:
        return f', '.join([
            f'{self.pattern.pattern}',
            f"repl='{self.repl_token}'",
        ])

    def process(self, datum: str, *args, **kwargs) -> str:
        return re.sub(self.pattern, self.repl_token, datum)


class ReplaceDigits(RegexSubs):
    def __init__(self, repl_token: str) -> None:
        super(ReplaceDigits, self).__init__(
            pattern=re.compile(r'\d+'),
            repl_token=repl_token,
        )
