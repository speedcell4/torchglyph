from collections import Counter
from typing import Any
from typing import List

from torchglyph.proc.abc import Flatten, PreProc


class ToInt(Flatten):
    def process(self, data: str, *args, **kwargs) -> int:
        return int(data)


class ToUpper(Flatten):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.upper()


class ToLower(Flatten):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.lower()


class ToCapitalized(Flatten):
    def process(self, data: str, *args, **kwargs) -> str:
        return data.capitalize()


class ToSub(Flatten):
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
