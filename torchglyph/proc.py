from collections import Counter
from typing import Optional, Union, Any, List

from torchglyph.vocab import Vocab


class Proc(object):
    def __add__(self, rhs: Optional[Union['Proc', 'Chain']]) -> Union['Proc', 'Chain']:
        if rhs is None:
            return self
        return Chain(self, rhs)

    def __radd__(self, lhs: Optional[Union['Proc', 'Chain']]) -> Union['Proc', 'Chain']:
        if lhs is None:
            return self
        return Chain(lhs, self)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return x


class Chain(Proc):
    def __init__(self, *procs: Optional[Union['Proc', 'Chain']]) -> None:
        super(Chain, self).__init__()
        self.procs = []
        for proc in procs:
            if proc is None:
                pass
            elif isinstance(proc, Proc):
                self.procs.append(proc)
            elif isinstance(proc, Chain):
                self.procs.extend(proc.procs)
            else:
                raise NotImplementedError(f'unsupported type :: {type(proc).__name__}')

    def __add__(self, rhs: Optional[Union['Proc', 'Chain']]) -> 'Chain':
        return Chain(*self.procs, rhs)

    def __radd__(self, lhs: Optional[Union['Proc', 'Chain']]) -> 'Chain':
        return Chain(lhs, *self.procs)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        for process in self.procs:
            x = process(x, *args, **kwargs)
        return x


# recursive processes

class RecurStrProc(Proc):
    def process(self, data: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if isinstance(data, str):
            return self.process(data, *args, **kwargs)
        return type(data)(self(datum, *args, **kwargs) for datum in data)


class RecurListStrProc(Proc):
    def process(self, data: List[str], *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if isinstance(data, list):
            assert len(data) > 0
            if isinstance(data[0], str):
                return self.process(data, *args, **kwargs)
        return type(data)(self(datum, *args, **kwargs) for datum in data)


# stage processes

class PreProc(Proc):
    def __call__(self, ins: Any, counter: Counter) -> Any:
        raise NotImplementedError


class VocabProc(Proc):
    def __call__(self, vocab: Union[Counter, Vocab]) -> Union[Counter, Vocab]:
        raise NotImplementedError


class PostProc(Proc):
    def __call__(self, ins: Any, vocab: Vocab) -> Any:
        raise NotImplementedError


class BatchProc(Proc):
    Batch = List[Any]

    def __call__(self, batch: Batch, vocab: Vocab) -> Any:
        raise NotImplementedError
