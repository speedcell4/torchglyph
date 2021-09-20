from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Any, List, Set, Tuple

__all__ = [
    'compress', 'subs',
    'Proc', 'Processors',
    'Identity', 'Lift', 'Chain',
    'Map', 'Filter',
]

Processors = Union[Optional['Proc'], List[Optional['Proc']]]


def compress(processors: Processors, allow_ellipsis: bool = True) -> List['Proc']:
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


def subs(processors: Processors, repl: Processors) -> Processors:
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

    def __add__(self, processors: Processors) -> 'Proc':
        return self.from_list([self] + compress(processors))

    def __radd__(self, processors: Processors) -> 'Proc':
        return self.from_list(compress(processors) + [self])

    @abstractmethod
    def __call__(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError


class Identity(Proc):
    def __repr__(self) -> str:
        return f'{None}'

    def __call__(self, data: Any, **kwargs) -> Any:
        return data


class Lift(Proc):
    Sequence = Union[Set[Any], List[Any], Tuple[Any, ...]]

    def __init__(self, proc: Proc) -> None:
        super(Lift, self).__init__()
        self.proc = proc

    def __repr__(self) -> str:
        return f'[{self.proc.__repr__()}]'

    def __call__(self, sequence: Sequence, **kwargs) -> Sequence:
        return type(sequence)([self.proc(item, **kwargs) for item in sequence])


class Chain(Proc):
    def __init__(self, processors: Processors) -> None:
        super(Chain, self).__init__()
        self.proc = compress(processors)

    def extra_repr(self) -> str:
        return ' + '.join([str(proc) for proc in self.proc])

    def __repr__(self) -> str:
        return self.extra_repr()

    def __add__(self, processors: Processors) -> 'Proc':
        return self.from_list(self.proc + compress(processors))

    def __radd__(self, processors: Processors) -> 'Proc':
        return self.from_list(compress(processors) + self.proc)

    def __call__(self, data: Any, **kwargs) -> Any:
        for proc in self.proc:
            data = proc(data, **kwargs)
        return data


class Map(Proc):
    Sequence = Union[Any, Set[Any], List[Any], Tuple[Any, ...]]

    def map(self, data: Any, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, sequence: Sequence, **kwargs) -> Sequence:
        if not isinstance(sequence, (set, list, tuple)):
            return self.map(sequence, **kwargs)
        return type(sequence)([self(data, **kwargs) for data in sequence])


class Filter(Proc):
    Sequence = Union[Any, Set[Any], List[Any], Tuple[Any, ...]]

    def filter(self, data: Any, **kwargs) -> bool:
        raise NotImplementedError

    def __call__(self, sequence: Sequence, **kwargs) -> Sequence:
        return type(sequence)([data for data in sequence if self.filter(data, **kwargs)])
