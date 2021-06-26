from typing import List, Any

from torchglyph.proc import Proc

__all__ = [
    'Flatten',
]


class Flatten(Proc):
    def __call__(self, data: List[List[Any]], **kwargs) -> List[Any]:
        return [item for datum in data for item in datum]
