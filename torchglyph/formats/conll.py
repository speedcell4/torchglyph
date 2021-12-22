from typing import Any, Type, Tuple, NamedTuple, IO, Iterable, get_type_hints

from torchglyph.formats.primitive import loads_type, dumps_type

__all__ = [
    'loads_token', 'iter_sentence',
    'dumps_token', 'dump_sentence',
]

Token = Tuple[Any, ...]
Sentence = Tuple[Tuple[Any, ...], ...]


def loads_token(s: str, *, config: Type[NamedTuple], sep: str = '\t') -> Token:
    return tuple(
        loads_type(s, tp=tp)
        for s, (name, tp) in zip(s.strip().split(sep=sep), get_type_hints(config).items())
        if not name.endswith('_')
    )


def iter_sentence(fp: IO, *, config: Type[NamedTuple], sep: str = '\t', blank: str = '') -> Iterable[Sentence]:
    sentence = []

    for s in fp:
        s = s.strip()
        if s != blank:
            sentence.append(loads_token(s, config=config, sep=sep))
        elif len(sentence) != 0:
            yield tuple(zip(*sentence))
            sentence = []

    if len(sentence) != 0:
        yield tuple(zip(*sentence))


def dumps_token(token: Token, sep: str = '\t') -> str:
    return sep.join(map(dumps_type, token))


def dump_sentence(sentences: Iterable[Sentence], fp: IO, *, sep: str = '\t', blank: str = '') -> None:
    for tokens in sentences:
        for token in tokens:
            fp.write(dumps_token(token, sep=sep))
            fp.write('\n')

        fp.write(blank)
        fp.write('\n')
