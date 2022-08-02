import gzip
import json
import logging
import pickle
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import List, Tuple

import requests
import torch
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


def cache_as_torch(method):
    def _cache_as_torch(self, path: Path, *args, **kwargs):
        cache = path.with_name(f'{path.name}.pt')
        cache.parent.mkdir(parents=True, exist_ok=True)

        if cache.exists():
            logger.info(f'loading from {cache}')
            obj = torch.load(cache, map_location=torch.device('cpu'))
        else:
            obj = method(self, path, *args, **kwargs)
            logger.info(f'saving to {cache}')
            torch.save(obj, f=cache, pickle_protocol=pickle.HIGHEST_PROTOCOL)

        return obj

    return _cache_as_torch


def cache_as_json(method):
    def _cache_as_json(self, path: Path, *args, **kwargs):
        cache = path.with_name(f'{path.name}.json')
        cache.parent.mkdir(parents=True, exist_ok=True)

        if cache.exists():
            logger.info(f'loading from {cache}')
            with cache.open(mode='r', encoding='utf-8') as fp:
                obj = json.load(fp)
        else:
            obj = method(self, path, *args, **kwargs)
            logger.info(f'saving to {cache}')
            with cache.open(mode='w', encoding='utf-8') as fp:
                json.dump(obj, fp=fp, indent=2, ensure_ascii=False)

        return obj

    return _cache_as_json


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
