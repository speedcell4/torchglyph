from pathlib import Path


def conllx_iter(path: Path, sep: str = '\t', encoding: str = 'utf-8'):
    sentence = []
    with path.open(mode='r', encoding=encoding) as fp:
        for row in fp:
            row = row.strip()
            if len(row) != 0:
                sentence.append(row.split(sep))
            elif len(sentence) != 0:
                yield sentence
                sentence = []

    if len(sentence) != 0:
        yield sentence
