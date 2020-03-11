from collections import Counter
from typing import List, Any, Union

from torchglyph.proc.abc import Proc, Recur
from torchglyph.proc.utilities import stoi
from torchglyph.vocab import Vocab


class GetLength(Proc):
    def __call__(self, ins: List[Any], **kwargs) -> int:
        return len(ins)


class GetRange(Proc):
    def __init__(self, reverse: bool) -> None:
        super(GetRange, self).__init__()
        self.reverse = reverse

    def __call__(self, ins: List[Any], **kwargs) -> List[int]:
        indices = range(len(ins))
        if self.reverse:
            indices = reversed(indices)
        return list(indices)


class GetMask(Proc):
    def __init__(self, mask_token: Union[str, int]) -> None:
        super(GetMask, self).__init__()
        self.token = mask_token

    def extra_repr(self) -> str:
        return f"'{self.token}'"

    def __call__(self, ins: List[Any], vocab: Vocab, **kwargs) -> List[int]:
        idx = stoi(token=self.token, vocab=vocab)
        return [idx for _ in ins]


class ToSub(Recur):
    def process(self, datum: Any, *args, **kwargs) -> List[Any]:
        return [sub for sub in datum]


class Prepend(Proc):
    def __init__(self, token: Any, num_repeats: int = 1) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def extra_repr(self) -> str:
        return f', '.join([
            f"'{self.token}'x{self.num_repeats}",
        ])

    def __call__(self, data: List[Any], counter: Counter) -> List[Any]:
        return [self.token for _ in range(self.num_repeats)] + list(data)


class Append(Proc):
    def __init__(self, token: Any, num_repeats: int = 1) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def extra_repr(self) -> str:
        return f', '.join([
            f"'{self.token}'x{self.num_repeats}",
        ])

    def __call__(self, data: List[Any], counter: Counter) -> List[Any]:
        return list(data) + [self.token for _ in range(self.num_repeats)]
