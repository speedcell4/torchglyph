from collections import Counter
from typing import List, Any, Union

from torchglyph.proc.abc import Proc, Flatten
from torchglyph.proc.utilities import stoi
from torchglyph.vocab import Vocab


class GetLength(Proc):
    def __call__(self, ins: List[Any], **kwargs) -> int:
        return len(ins)


class GetRange(Proc):
    def __call__(self, ins: List[Any], **kwargs) -> List[int]:
        return [idx for idx in range(len(ins))]


class GetMask(Proc):
    def __init__(self, mask_token: Union[str, int]) -> None:
        super(GetMask, self).__init__()
        self.mask_token = mask_token

    def __call__(self, ins: List[Any], vocab: Vocab, **kwargs) -> List[int]:
        mask_idx = stoi(token=self.mask_token, vocab=vocab)
        return [mask_idx for _ in ins]


class ToSub(Flatten):
    def process(self, data: Any, *args, **kwargs) -> List[Any]:
        return [c for c in data]


class Prepend(Proc):
    def __init__(self, token: Any, num_repeats: int = 1) -> None:
        super(Prepend, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def __call__(self, data: List[Any], counter: Counter) -> List[Any]:
        return [self.token for _ in range(self.num_repeats)] + list(data)


class Append(Proc):
    def __init__(self, token: Any, num_repeats: int = 1) -> None:
        super(Append, self).__init__()
        self.token = token
        self.num_repeats = num_repeats

    def __call__(self, data: List[Any], counter: Counter) -> List[Any]:
        return list(data) + [self.token for _ in range(self.num_repeats)]
