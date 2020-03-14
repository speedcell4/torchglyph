import re
from typing import Pattern, List

from torchglyph.proc.abc import Recur
from torchglyph.vocab import Vocab


class ToInt(Recur):
    def process(self, data: str, *args, **kwargs) -> int:
        return int(data)


class ToUpper(Recur):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.upper()


class ToLower(Recur):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.lower()


class ToCapitalized(Recur):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.capitalize()


class ToSubList(Recur):
    def process(self, data: str, *args, **kwargs) -> List[str]:
        return [sub for sub in data]


class ReplacePattern(Recur):
    def __init__(self, pattern: Pattern, repl_token: str) -> None:
        super(ReplacePattern, self).__init__()
        self.pattern = pattern
        self.repl_token = repl_token

    def extra_repr(self) -> str:
        return f', '.join([
            f'{self.pattern.pattern}',
            f"repl='{self.repl_token}'",
        ])

    def process(self, data: str, *args, **kwargs) -> str:
        return re.sub(self.pattern, self.repl_token, data)


class ReplaceDigits(ReplacePattern):
    def __init__(self, repl_token: str) -> None:
        super(ReplaceDigits, self).__init__(
            pattern=re.compile(r'\d+'),
            repl_token=repl_token,
        )


class Numbering(Recur):
    def process(self, datum: str, vocab: Vocab, **kwargs) -> int:
        return vocab.stoi[datum]
