from typing import Any, Set, List, Tuple, Union

from torchglyph.proc import Proc

Sequence = Union[Set[Any], List[Any], Tuple[Any, ...]]


class ToLen(Proc):
    def __call__(self, sequence: Sequence, **kwargs) -> int:
        return len(sequence)


class Prepend(Proc):
    def __init__(self, item: Any) -> None:
        super(Prepend, self).__init__()
        self.item = item

    def extra_repr(self) -> str:
        return f'[{self.item}]'

    def __call__(self, sequence: Sequence, **kwargs) -> Sequence:
        return type(sequence)([self.item] + list(sequence))


class Append(Proc):
    def __init__(self, item: Any) -> None:
        super(Append, self).__init__()
        self.item = item

    def extra_repr(self) -> str:
        return f'[{self.item}]'

    def __call__(self, sequence: Sequence, **kwargs) -> Sequence:
        return type(sequence)(list(sequence) + [self.item])
