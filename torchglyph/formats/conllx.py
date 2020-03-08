from typing import Iterable, Any, Tuple

from torchglyph.io import IO, open_io

Sentence = Iterable[Tuple[Any, ...]]
RawSentence = Iterable[Tuple[str, ...]]


def load(f: IO, *, sep: str = '\t', mode: str = 'r', encoding: str = 'utf-8') -> Iterable[RawSentence]:
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


def dumps(sentence: Sentence, *, sep: str = '\t') -> Iterable[str]:
    for tokens in sentence:
        yield sep.join(f'{t}' for t in tokens)


def dump(sentence: Sentence, f: IO, *, sep: str = '\t', mode: str = 'w', encoding: str = 'utf-8') -> None:
    with open_io(f, mode=mode, encoding=encoding) as fp:
        for raw in dumps(sentence, sep=sep):
            print(raw, file=fp)
        print('', file=fp)
