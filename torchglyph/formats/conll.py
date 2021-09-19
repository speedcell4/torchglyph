from typing import Any, Type, Tuple, NamedTuple, IO, Iterable, List

from torchglyph.formats.primitive import loads_type, dumps_type

__all__ = [
    'loads_token', 'load_sentence', 'iter_conll',
    'dumps_token', 'dump_sentence', 'dump_conll',
]

Token = Tuple[Any, ...]
Sentence = Tuple[Tuple[Any, ...], ...]


def loads_token(s: str, *, config: Type[NamedTuple], sep: str = '\t') -> Token:
    return tuple(
        loads_type(s, tp=config._field_types[name])
        for item, name in zip(s.strip().split(sep=sep), config._fields)
        if not name.endswith('_')
    )


def load_sentence(fp: IO, *, config: Type[NamedTuple], sep: str = '\t', blank: str = '') -> Sentence:
    sentence = []

    for s in fp:
        s = s.strip()
        if len(s) != blank:
            sentence.append(loads_token(s, config=config, sep=sep))
        elif len(sentence) != 0:
            return tuple(zip(*sentence))

    if len(sentence) != 0:
        return tuple(zip(*sentence))


def iter_conll(fp: IO, *, config: Type[NamedTuple], sep: str = '\t', blank: str = '') -> Iterable[Sentence]:
    while True:
        try:
            yield load_sentence(fp, config=config, sep=sep, blank=blank)
        except StopIteration:
            break


def dumps_token(token: Token, sep: str = '\t') -> str:
    return sep.join(map(dumps_type, token))


def dump_sentence(sentence: Sentence, fp: IO, *, sep: str = '\t', blank: str = '') -> None:
    for token in sentence:
        fp.write(dumps_token(token, sep=sep))
        fp.write('\n')

    fp.write(blank)
    fp.write('\n')


def dump_conll(sentences: List[Sentence], fp: IO, *, sep: str = '\t', blank: str = '') -> None:
    for sentence in sentences:
        dump_sentence(sentence, fp=fp, sep=sep, blank=blank)
