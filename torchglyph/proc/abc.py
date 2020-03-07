from typing import Optional, Union, Any, List, Callable, Tuple

from torchglyph.vocab import Vocab

PaLP = Union[Optional['Proc'], List[Optional['Proc']]]


def compress(procs: PaLP, allow_ellipsis: bool = False) -> List['Proc']:
    if procs is None or isinstance(procs, Identity):
        return []
    if procs is ...:
        if allow_ellipsis:
            return [...]
        else:
            raise ValueError(f'ellipsis is not allowed here')
    if isinstance(procs, Chain):
        return procs.procs
    if isinstance(procs, Proc):
        return [procs]
    return [x for proc in procs for x in compress(proc, allow_ellipsis=allow_ellipsis)]


def subs(procs: PaLP, repl: 'Proc') -> PaLP:
    return [repl if proc is ... else proc for proc in compress(procs, allow_ellipsis=True)]


class Proc(object):
    @classmethod
    def from_list(cls, procs: List['Proc']) -> 'Proc':
        if len(procs) == 0:
            return Identity()
        if len(procs) == 1:
            return procs[0]
        return Chain(procs)

    def __add__(self, rhs: PaLP) -> 'Proc':
        return self.from_list([self] + compress(rhs))

    def __radd__(self, lhs: PaLP) -> 'Proc':
        return self.from_list(compress(lhs) + [self])

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        raise NotImplementedError


class Identity(Proc):
    def __call__(self, x: Any, *args, **kwargs) -> Any:
        return x


class Chain(Proc):
    def __init__(self, procs: PaLP) -> None:
        super(Chain, self).__init__()
        self.procs = compress(procs)

    def __add__(self, rhs: PaLP) -> 'Proc':
        return self.from_list(self.procs + compress(rhs))

    def __radd__(self, lhs: PaLP) -> 'Proc':
        return self.from_list(compress(lhs) + self.procs)

    def __call__(self, x: Any, *args, **kwargs) -> Any:
        for process in self.procs:
            x = process(x, *args, **kwargs)
        return x


class Lift(Proc):
    def __init__(self, proc: Proc) -> None:
        super(Lift, self).__init__()
        self.proc = proc

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        return type(data)([self.proc(datum, *args, **kwargs) for datum in data])


class Recur(Proc):
    def process(self, datum: str, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Any, *args, **kwargs) -> Any:
        if isinstance(data, str):
            return self.process(data, *args, **kwargs)
        return type(data)([self(datum, *args, **kwargs) for datum in data])


class ScanL(Proc):
    def __init__(self, fn: Callable[[Any, Any], Tuple[Any, Any]], init: Any) -> None:
        super(ScanL, self).__init__()
        self.fn = fn
        self.init = init

    def __call__(self, xs: List[Any], vocab: Vocab = None) -> List[Any]:
        z = self.init

        ys = []
        for x in xs:
            y, z = self.fn(x, z)
            ys.append(y)
        return type(xs)(ys)


class ScanR(Proc):
    def __init__(self, fn: Callable[[Any, Any], Tuple[Any, Any]], init: Any) -> None:
        super(ScanR, self).__init__()
        self.fn = fn
        self.init = init

    def __call__(self, xs: List[Any], vocab: Vocab = None) -> List[Any]:
        z = self.init

        ys = []
        for x in xs:
            z, y = self.fn(z, x)
            ys.append(y)
        return type(xs)(ys)
