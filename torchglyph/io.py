import logging
import os
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm


# copied and modified from https://github.com/pytorch/text
def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None) -> None:
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return inner


# copied and modified from https://github.com/pytorch/text
def download_and_unzip(url: str, dest: Path) -> None:
    if not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=f'downloading {dest}') as t:
        try:
            urlretrieve(url, str(dest), reporthook=reporthook(t))
        except KeyboardInterrupt as err:  # remove the partial zip file
            os.remove(str(dest))
            raise err

    if dest.suffix == '.zip':
        logging.info(f'extracting {dest}')
        with zipfile.ZipFile(dest, "r") as zf:
            zf.extractall(path=dest.parent)
    elif dest.suffixes[:-2] == ['.tar', '.gz']:
        logging.info(f'extracting {dest}')
        with tarfile.open(dest, 'r:gz') as tar:
            tar.extractall(path=dest.parent)


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
