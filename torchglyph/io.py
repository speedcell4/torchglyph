import gzip
import logging
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Union, Pattern
from urllib.request import urlretrieve

from tqdm import tqdm

logger = logging.getLogger(__name__)


def download_and_unzip(url: str, path: Path) -> Path:
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    with tqdm(unit='B', unit_scale=True, miniters=1, desc=f'downloading {path}') as t:
        try:
            urlretrieve(url, str(path), reporthook=reporthook(t))
        except KeyboardInterrupt as err:  # remove the partial zip file
            os.remove(str(path))
            raise err

    return extract_files(path)


def extract_files(path: Path) -> Path:
    if path.suffix == '.zip':
        logger.info(f'extracting {path}')
        with zipfile.ZipFile(path, "r") as fp:
            fp.extractall(path=path.parent)
    elif path.suffixes[-2:] == ['.tar', '.gz']:
        logger.info(f'extracting {path}')
        with tarfile.open(path, 'r:gz') as fp:
            fp.extractall(path=path.parent)
    elif path.suffix == '.gz':
        logger.info(f'extracting {path}')
        with gzip.open(path, mode='rb') as fs:
            with path.with_suffix('').open(mode='wb') as fd:
                shutil.copyfileobj(fs, fd)
    return path


def toggle_loggers(pattern: Union[str, Pattern], enable: bool) -> None:
    for name in logging.root.manager.loggerDict:  # type:str
        if re.match(pattern, name) is not None:
            logging.getLogger(name).disabled = not enable
