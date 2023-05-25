from typing import Any, IO, Iterable, NamedTuple, Tuple, Type, get_type_hints

from torchglyph.formats.primitive import dumps_type, loads_type

Token = Tuple[Any, ...]
Sentence = Tuple[Tuple[Any, ...], ...]


def loads_token(string: str, *, config: Type[NamedTuple], sep: str = '\t') -> Token:
    return tuple(
        loads_type(s, tp=tp)
        for s, (name, tp) in zip(string.strip().split(sep=sep), get_type_hints(config).items())
        if not name.endswith('_')
    )


def iter_sentence(fp: IO, *, config: Type[NamedTuple], sep: str = '\t', blank: str = '') -> Iterable[Sentence]:
    sentence = []

    for string in fp:
        string = string.strip()
        if string != blank:
            sentence.append(loads_token(string, config=config, sep=sep))
        elif len(sentence) != 0:
            yield tuple(zip(*sentence))
            sentence = []

    if len(sentence) != 0:
        yield tuple(zip(*sentence))


def dumps_token(token: Token, *, config: Type[NamedTuple], sep: str = '\t') -> str:
    return sep.join([
        dumps_type(data)
        for data, (name, tp) in zip(token, get_type_hints(config).items())
        if not name.endswith('_')
    ])


def dump_sentence(sentence: Sentence, fp: IO, *, config: Type[NamedTuple], sep: str = '\t', blank: str = '') -> None:
    for token in sentence:
        print(dumps_token(token, config=config, sep=sep), file=fp)
    print(blank, file=fp)
