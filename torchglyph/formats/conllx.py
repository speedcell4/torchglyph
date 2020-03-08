from typing import Iterable, List, Any, Tuple

from torchglyph.io import IO, open_io

Sentence = List[Tuple[Any]]


def load(f: IO, *, sep: str = '\t', mode: str = 'r', encoding: str = 'utf-8') -> Iterable[Sentence]:
    sent = []

    with open_io(f, mode=mode, encoding=encoding) as fp:
        for raw in fp:
            raw = raw.strip()
            if len(raw) != 0:
                sent.append(raw.split(sep))
            elif len(sent) != 0:
                yield sent
                sent = []

        if len(sent) != 0:
            yield sent


def dumps(obj: Sentence, *, sep: str = '\t') -> Iterable[str]:
    for tokens in obj:
        yield sep.join(f'{t}' for t in tokens)


def dump(obj: Sentence, f: IO, *, sep: str = '\t', mode: str = 'w', encoding: str = 'utf-8') -> None:
    with open_io(f, mode=mode, encoding=encoding) as fp:
        for raw in dumps(obj, sep=sep):
            print(raw, file=fp)
        print(raw, file=fp)
