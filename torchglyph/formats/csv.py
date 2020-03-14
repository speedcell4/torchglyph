import csv
from io import StringIO
from typing import Iterable, Union
from typing import Tuple, IO

from torchglyph.io import open_io

Sentence = RawSentence = Tuple[Union[str, int]]


def load(f: IO, *, sep=',', quotechar='"', mode: str = 'r', encoding: str = 'utf-8') -> Iterable[RawSentence]:
    with open_io(f, mode=mode, encoding=encoding) as fp:
        yield from csv.reader(fp, delimiter=sep, quotechar=quotechar)


def dumps(sentence: Sentence, *, sep=',', quotechar='"') -> str:
    with StringIO(initial_value='') as s:
        fp = csv.writer(s, delimiter=sep, quotechar=quotechar)
        fp.writerow(sentence)
        return s.getvalue()


def dump(sentence: Sentence, f: IO, *, sep=',', quotechar='"', mode: str = 'w', encoding: str = 'utf-8') -> None:
    with open_io(f, mode=mode, encoding=encoding) as fp:
        print(dumps(sentence, sep=sep, quotechar=quotechar), file=fp)
