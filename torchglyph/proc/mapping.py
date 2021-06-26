import re
from typing import Union, List, Tuple, Set, Pattern, Any

from torchglyph.proc.abc import Proc
from torchglyph.vocab import Vocab

__all__ = [
    'Map',
    'Numbering',
    'ToInt', 'ToBool', 'ToFloat',
    'ToLower', 'ToUpper', 'ToCapitalized',
    'Regex',
]


class Map(Proc):
    def map(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Union[Any, Set[Any], List[Any], Tuple[Any, ...]], **kwargs):
        if not isinstance(data, (set, list, tuple)):
            return self.map(data, **kwargs)
        return type(data)([self(datum, **kwargs) for datum in data])


class Numbering(Map):
    def map(self, data: str, *, vocab: Vocab, **kwargs) -> int:
        return vocab.stoi[data]


class ToInt(Map):
    def map(self, data: str, **kwargs) -> int:
        return int(data)


class ToBool(Map):
    def map(self, data: str, **kwargs) -> bool:
        return bool(data)


class ToFloat(Map):
    def map(self, data: str, **kwargs) -> float:
        return float(data)


class ToLower(Map):
    def map(self, data: str, **kwargs) -> str:
        return data.lower()


class ToUpper(Map):
    def map(self, data: str, **kwargs) -> str:
        return data.upper()


class ToCapitalized(Map):
    def map(self, data: str, **kwargs) -> str:
        return data.capitalize()


class Regex(Map):
    def __init__(self, pattern: Pattern, repl: str) -> None:
        super(Regex, self).__init__()
        self.pattern = pattern
        self.repl = repl

    def extra_repr(self) -> str:
        return f'{self.pattern} -> {self.repl}'

    def map(self, data: str, **kwargs) -> str:
        return re.sub(pattern=self.pattern, repl=self.repl, string=data)
