from collections import Counter
from typing import Any
from typing import List

from torchglyph.proc import RecurStrProc, RecurListStrProc, PreProc


class ToInt(RecurStrProc):
    def process(self, data: str, *args, **kwargs) -> int:
        return int(data)


class ToUpper(RecurStrProc):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.upper()


class ToLower(RecurStrProc):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.lower()


class ToCapitalized(RecurStrProc):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.capitalize()


class ToChar(RecurStrProc):
    def process(self, data: str, *args, **kwargs) -> List[str]:
        return [c for c in data]


class Prepend(PreProc):
    def __init__(self, token: Any, num_repeats: int = 1) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def __call__(self, ins: List[Any], counter: Counter) -> List:
        return [self.token for _ in range(self.num_repeats)] + list(ins)


class Append(PreProc):
    def __init__(self, token: Any, num_repeats: int = 1) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def __call__(self, ins: List[Any], counter: Counter) -> List:
        return list(ins) + [self.token for _ in range(self.num_repeats)]


class ToMask(RecurStrProc):
    def __init__(self, token: Any) -> None:
        self.token = token

    def process(self, data: str, *args, **kwargs) -> Any:
        return self.token


class ToRange(RecurListStrProc):
    def process(self, data: List[str], *args, **kwargs) -> List[int]:
        return list(range(len(data)))


class AddToCounter(PreProc):
    @classmethod
    def obtain_tokens(cls, ins):
        if isinstance(ins, str):
            yield ins
        else:
            for x in ins:
                yield from cls.obtain_tokens(x)

    def __call__(self, ins, counter: Counter) -> Any:
        counter.update(self.obtain_tokens(ins))
        return ins
