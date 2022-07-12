from typing import Any

from torchglyph.proc import Proc


class ToLen(Proc):
    def __call__(self, data: Any, **kwargs) -> int:
        return len(data)


class Prepend(Proc):
    def __init__(self, item: Any) -> None:
        super(Prepend, self).__init__()
        self.item = item

    def extra_repr(self) -> str:
        return f'[{self.item}]'

    def __call__(self, data: Any, **kwargs) -> Any:
        return type(data)(list(data) + [self.item])


class Append(Proc):
    def __init__(self, item: Any) -> None:
        super(Append, self).__init__()
        self.item = item

    def extra_repr(self) -> str:
        return f'[{self.item}]'

    def __call__(self, data: Any, **kwargs) -> Any:
        return type(data)([self.item] + list(data))
