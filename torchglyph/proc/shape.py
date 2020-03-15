from collections import Counter
from typing import List, Any, Union

from torchglyph.proc import Proc
from torchglyph.proc import stoi
from torchglyph.vocab import Vocab


class GetLength(Proc):
    def __call__(self, data: List[Any], *args, **kwargs) -> int:
        return len(data)


class GetRange(Proc):
    def __init__(self, reverse: bool) -> None:
        super(GetRange, self).__init__()
        self.reverse = reverse

    def extra_repr(self) -> str:
        if not self.reverse:
            return 'ascending'
        else:
            return 'descending'

    def __call__(self, data: List[Any], *args, **kwargs) -> List[int]:
        indices = range(len(data))
        if self.reverse:
            indices = reversed(indices)
        return list(indices)


class GetMask(Proc):
    def __init__(self, token: Union[str, int]) -> None:
        super(GetMask, self).__init__()
        self.token = token

    def extra_repr(self) -> str:
        return f"'{self.token}'"

    def __call__(self, data: List[Any], vocab: Vocab, *args, **kwargs) -> List[int]:
        token = stoi(token=self.token, vocab=vocab)
        return [token for _ in data]


class Prepend(Proc):
    def __init__(self, token: Any, num_repeats: int) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def extra_repr(self) -> str:
        return ', '.join([
            f"'{self.token}'x{self.num_repeats}",
        ])

    def __call__(self, data: List[Any], counter: Counter, *args, **kwargs) -> List[Any]:
        return [self.token for _ in range(self.num_repeats)] + list(data)


class Append(Proc):
    def __init__(self, token: Any, num_repeats: int) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def extra_repr(self) -> str:
        return ', '.join([
            f"'{self.token}'x{self.num_repeats}",
        ])

    def __call__(self, data: List[Any], counter: Counter, *args, **kwargs) -> List[Any]:
        return list(data) + [self.token for _ in range(self.num_repeats)]
