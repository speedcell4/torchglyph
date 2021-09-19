from typing import IO, Iterable, Tuple, Sequence, List

__all__ = [
    'load_freq', 'loads_freq', 'iter_freq',
    'dump_freq', 'dumps_freq',
]


def loads_freq(s: str, *, sep: str = '\t') -> Tuple[str, int]:
    token, freq = s.strip().split(sep=sep)
    return str(token), int(freq)


def load_freq(fp: IO, *, sep: str = '\t') -> List[Tuple[str, int]]:
    return [loads_freq(s, sep=sep) for s in fp]


def iter_freq(fp: IO, *, sep: str = '\t') -> Iterable[Tuple[str, int]]:
    yield from map(lambda s: loads_freq(s, sep=sep), fp)


def dumps_freq(token: str, freq: int, *, sep: str = '\t') -> str:
    return f'{token}{sep}{freq}'


def dump_freq(obj: Sequence[Tuple[str, int]], fp: IO, *, sep: str = '\t') -> None:
    for token, freq in obj:
        fp.write(dumps_freq(token, freq, sep=sep))
        fp.write('\n')
