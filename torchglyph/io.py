import gzip
import logging
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests
from requests import Response
from tqdm import tqdm

from torchglyph import data_dir

logger = logging.getLogger(__name__)


def download(url: str, path: Path, exist_ok: bool = True, chunk_size: int = 1 << 20) -> Path:
    response: Response = requests.get(url=url, stream=True)
    response.raise_for_status()

    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or not exist_ok:
        with path.open(mode='wb') as fp:
            for content in tqdm(response.iter_content(chunk_size=chunk_size),
                                total=int(response.headers['Content-Length']) // chunk_size,
                                desc=f'downloading from {url}', unit='MB'):
                fp.write(content)

    return path


def unzip(path: Path) -> Path:
    logger.info(f'extracting files from {path}')

    if path.suffix == '.zip':
        with zipfile.ZipFile(path, "r") as fp:
            fp.extractall(path=path.parent)
    elif path.suffixes[-2:] == ['.tar', '.gz']:
        with tarfile.open(path, 'r:gz') as fp:
            fp.extractall(path=path.parent)
    elif path.suffix == '.gz':
        with gzip.open(path, mode='rb') as fsrc:
            with path.with_suffix('').open(mode='wb') as fdst:
                shutil.copyfileobj(fsrc, fdst)

    return path


class DownloadMixin(object):
    name: str

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        raise NotImplementedError

    @classmethod
    def paths(cls, root: Path = data_dir, **kwargs) -> List[Path]:
        root = root / getattr(cls, 'name', cls.__name__).lower()

        paths = []
        for url, path, *names in cls.urls(**kwargs):
            if len(names) == 0:
                names = [path]
            if any(not (root / name).exists() for name in names):
                unzip(path=download(url=url, path=root / path, exist_ok=False))
            for name in names:
                if not (root / name).exists():
                    raise FileNotFoundError(f'{root / name} is not obtainable from {url}')
                paths.append(root / name)

        return paths
