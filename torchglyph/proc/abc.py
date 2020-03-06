from collections import Counter
from typing import Optional, Union, Any, List, Callable, Tuple

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
        raise NotImplementedError


class Identity(Proc):
    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return x


class Lift(Proc):
    def __init__(self, proc: Proc) -> None:
        super(Lift, self).__init__()
        self.proc = proc

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        return type(data)([self.proc(datum, *args, **kwargs) for datum in data])


class Flatten(Proc):
    def process(self, data: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if isinstance(data, str):
            return self.process(data, *args, **kwargs)
        return type(data)([self(datum, *args, **kwargs) for datum in data])


class Scan(Proc):
    def __init__(self, fn: Callable[[Any, Any], Tuple[Any, Any]], init: Any) -> None:
        super(Scan, self).__init__()
        self.fn = fn
        self.init = init

    def __call__(self, xs: List[Any], vocab: Vocab = None) -> List[Any]:
        z = self.init

        ys = []
        for x in xs:
            y, z = self.fn(x, z)
            ys.append(y)
        return ys


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
