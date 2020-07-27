import gzip
import logging
import os
import re
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Union, TextIO, Pattern
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)

IO = Union[str, Path, TextIO]


@contextmanager
def open_io(f: IO, mode: str, encoding: str):
    try:
        if isinstance(f, (str, Path)):
            with open(f, mode=mode, encoding=encoding) as fp:
                yield fp
        else:
            yield f
    finally:
        pass


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
def download_and_unzip(url: str, dest: Path) -> Path:
    if dest.exists():
        return dest

    if not dest.parent.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(unit='B', unit_scale=True, miniters=1, desc=f'downloading {dest}') as t:
        try:
            urlretrieve(url, str(dest), reporthook=reporthook(t))
        except KeyboardInterrupt as err:  # remove the partial zip file
            os.remove(str(dest))
            raise err

    if dest.suffix == '.zip':
        logger.info(f'extracting {dest}')
        with zipfile.ZipFile(dest, "r") as fp:
            fp.extractall(path=dest.parent)
    elif dest.suffixes[-2:] == ['.tar', '.gz']:
        logger.info(f'extracting {dest}')
        with tarfile.open(dest, 'r:gz') as fp:
            fp.extractall(path=dest.parent)
    elif dest.suffix == '.gz':
        logger.info(f'extracting {dest}')
        with gzip.open(dest, mode='rb') as fs:
            with dest.with_suffix('').open(mode='wb') as fd:
                shutil.copyfileobj(fs, fd)

    return dest


def toggle_loggers(pattern: Union[str, Pattern], enable: bool) -> None:
    for name in logging.root.manager.loggerDict:  # type:str
        if re.match(pattern, name) is not None:
            logging.getLogger(name).disabled = not enable
