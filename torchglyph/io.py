import gzip
import logging
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Union, Pattern, List, Tuple

import requests
from requests import Response
from tqdm import tqdm

from torchglyph import data_dir

logger = logging.getLogger(__name__)

__all__ = [
    'toggle_loggers',
    'DownloadMixin', 'download', 'extract',
]


def toggle_loggers(pattern: Union[str, Pattern], enable: bool) -> None:
    for name in logging.root.manager.loggerDict:  # type:str
        if re.match(pattern, name) is not None:
            logging.getLogger(name).disabled = not enable


def download(url: str, path: Path, exist_ok: bool = True, chunk_size: int = 1024 * 1024) -> Path:
    path.mkdir(parents=True, exist_ok=True)

    response: Response = requests.get(url=url, stream=True)
    assert response.status_code == 200, f'{response.status_code} != {200}'

    if not path.exists() or not exist_ok:
        with path.open(mode='wb') as fp:
            for chunk in tqdm(response.iter_content(chunk_size=chunk_size),
                              total=int(response.headers['Content-Length']) / chunk_size,
                              desc=f'downloading from {url}', unit='MB'):
                fp.write(chunk)

    return path


def extract(path: Path) -> Path:
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


class DownloadMixin(object):
    name: str
    url: List[Tuple[str, ...]]

    def urls(self, **kwargs) -> List[Tuple[str, ...]]:
        return self.url

    def paths(self, root: Path = data_dir, **kwargs) -> List[Path]:
        cls = self.__class__
        root = root / getattr(cls, 'name', cls.__name__).lower()

        paths = []
        for url, path, *names in self.urls(**kwargs):
            if len(names) == 0:
                names = [path]
            if any(not (root / name).exists() for name in names):
                extract(path=download(url=url, path=root / path, exist_ok=False))
            for name in names:
                if not (root / name).exists():
                    raise FileNotFoundError(f'{root / name} is not obtainable from {url}')
                paths.append(root / name)

        return paths
