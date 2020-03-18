import re
from typing import Pattern, List

from torchglyph.proc import RecurStr
from torchglyph.vocab import Vocab


class Numbering(RecurStr):
    def process(self, data: str, vocab: Vocab, **kwargs) -> int:
        return vocab.stoi[data]


class ToInt(RecurStr):
    def process(self, data: str, **kwargs) -> int:
        return int(data)


class ToUpper(RecurStr):
    def process(self, data: str, **kwargs) -> str:
        return data.upper()


class ToLower(RecurStr):
    def process(self, data: str, **kwargs) -> str:
        return data.lower()


class ToCapitalized(RecurStr):
    def process(self, data: str, **kwargs) -> str:
        return data.capitalize()


class ToSubList(RecurStr):
    def process(self, data: str, **kwargs) -> List[str]:
        return [sub for sub in data]


class ReplacePattern(RecurStr):
    def __init__(self, pattern: Pattern, repl_token: str) -> None:
        super(ReplacePattern, self).__init__()
        self.pattern = pattern
        self.repl_token = repl_token

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.pattern.pattern}',
            f"repl='{self.repl_token}'",
        ])

    def process(self, data: str, **kwargs) -> str:
        return re.sub(self.pattern, self.repl_token, data)


class ReplaceDigits(ReplacePattern):
    def __init__(self, repl_token: str) -> None:
        super(ReplaceDigits, self).__init__(
            pattern=re.compile(r'\d+'),
            repl_token=repl_token,
        )
