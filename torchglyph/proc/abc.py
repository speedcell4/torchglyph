from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any, List, Set, Tuple

from torch import Tensor

__all__ = [
    'compress', 'subs',
    'Proc', 'Identity', 'Lift',
    'ProcList', 'Chain',
    'Map', 'Filter',
]

ProcList = Union[Optional['Proc'], List[Optional['Proc']]]


def compress(processors: ProcList, allow_ellipsis: bool = True) -> List['Proc']:
    if processors is None or isinstance(processors, Identity):
        return []
    if processors is ...:
        if allow_ellipsis:
            return [...]
        else:
            raise ValueError(f'ellipsis is not allowed here, {processors}')
    if isinstance(processors, Chain):
        return processors.proc
    if isinstance(processors, Proc):
        return [processors]
    return [p for proc in processors for p in compress(proc, allow_ellipsis=allow_ellipsis)]


def subs(processors: ProcList, repl: ProcList) -> ProcList:
    return [repl if proc is ... else proc for proc in compress(processors, allow_ellipsis=True)]


class Proc(object, metaclass=ABCMeta):
    @classmethod
    def from_list(cls, processors: List['Proc']) -> 'Proc':
        if len(processors) == 0:
            return Identity()
        if len(processors) == 1:
            return processors[0]
        return Chain(processors)

    def extra_repr(self) -> str:
        return f''

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def __add__(self, other: ProcList) -> 'Proc':
        return self.from_list([self] + compress(other))

    def __radd__(self, other: ProcList) -> 'Proc':
        return self.from_list(compress(other) + [self])

    @abstractmethod
    def __call__(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError


class Identity(Proc):
    def __repr__(self) -> str:
        return f'{None}'

    def __call__(self, data: Any, **kwargs) -> Any:
        return data


class Lift(Proc):
    def __init__(self, proc: Proc) -> None:
        super(Lift, self).__init__()
        self.proc = proc

    def __repr__(self) -> str:
        return f'[{self.proc.__repr__()}]'

    def __call__(self, data: Any, **kwargs) -> Any:
        return type(data)([self.proc(datum, **kwargs) for datum in data])


class Chain(Proc):
    def __init__(self, processors: ProcList) -> None:
        super(Chain, self).__init__()
        self.proc = compress(processors)

    def extra_repr(self) -> str:
        return ' + '.join([str(proc) for proc in self.proc])

    def __repr__(self) -> str:
        return f'{self.extra_repr()}'

    def __add__(self, other: ProcList) -> 'Proc':
        return self.from_list(self.proc + compress(other))

    def __radd__(self, other: ProcList) -> 'Proc':
        return self.from_list(compress(other) + self.proc)

    def __call__(self, data: Any, **kwargs) -> Any:
        for proc in self.proc:
            data = proc(data, **kwargs)
        return data


class Map(Proc):
    Container = Union[Set[Tensor], List[Tensor], Tuple[Tensor, ...]]

    def map(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, data: Union[Any, Container], **kwargs) -> Union[Any, Container]:
        if not isinstance(data, (set, list, tuple)):
            return self.map(data, **kwargs)
        return type(data)([self(datum, **kwargs) for datum in data])


class Filter(Proc):
    Container = Union[Set[Tensor], List[Tensor], Tuple[Tensor, ...]]

    def predicate(self, data: Any, **kwargs) -> bool:
        raise NotImplementedError

    def __call__(self, data: Container, **kwargs) -> Container:
        return type(data)([datum for datum in data if self.predicate(datum, **kwargs)])
