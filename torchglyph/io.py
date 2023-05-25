import gzip
import json
import logging
import shutil
import tarfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, List, Tuple

import torch
from datasets.config import DATASETDICT_JSON_FILENAME, DATASET_INFO_FILENAME
from datasets.download import DownloadConfig, DownloadManager
from datasets.fingerprint import Hasher
from filelock import FileLock
from torch import nn

from torchglyph import DEBUG, data_dir

logger = logging.getLogger(__name__)

ARGS_JSON = 'args.json'
SOTA_JSON = 'sota.json'
CHECKPOINT_PT = 'checkpoint.pt'


def download_and_extract(url: str, name: str, root: Path = data_dir) -> Path:
    manager = DownloadManager(
        dataset_name=name,
        download_config=DownloadConfig(
            cache_dir=root / name,
            download_desc=f'Downloading {url}',
        ),
    )

    return Path(manager.download_and_extract(url))


def hf_hash(**kwargs) -> str:
    hasher = Hasher()

    for key, value in sorted(kwargs.items()):
        hasher.update(key)
        hasher.update(value)

    return hasher.hexdigest()


def cache_file(path: Path, **kwargs) -> Path:
    cache = path.resolve()
    cache.parent.mkdir(parents=True, exist_ok=True)
    return cache.parent / f'{cache.name}.{hf_hash(__torchglyph=DEBUG, **kwargs)}'


def cache_folder(path: Path, **kwargs) -> Path:
    cache = path.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    return cache / hf_hash(__torchglyph=DEBUG, **kwargs)


def all_exits(path: Path, *names: str) -> bool:
    for name in names:
        if not (path / name).exists():
            return False

    return True


def is_dataset_folder(path: Path) -> bool:
    path = path / DATASET_INFO_FILENAME
    return path.is_file() and path.exists()


def is_dataset_dict_folder(path: Path) -> bool:
    path = path / DATASETDICT_JSON_FILENAME
    return path.is_file() and path.exists()


@contextmanager
def lock_folder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.resolve() / '.lock')):
        yield


def load_json(path: Path) -> Any:
    with path.open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


def load_args(out_dir: Path, name: str = ARGS_JSON) -> Any:
    return load_json(path=out_dir / name)


def load_sota(out_dir: Path, name: str = SOTA_JSON) -> Any:
    return load_json(path=out_dir / name)


def save_json(path: Path, **kwargs) -> None:
    data = {}
    if not path.exists():
        logger.info(f'saving to {path}')
    else:
        with path.open(mode='r', encoding='utf-8') as fp:
            data = json.load(fp=fp)

    with path.open(mode='w', encoding='utf-8') as fp:
        json.dump({**data, **kwargs}, fp=fp, indent=2, ensure_ascii=False)


def save_args(out_dir: Path, name: str = ARGS_JSON, **kwargs) -> None:
    return save_json(path=out_dir / name, **kwargs)


def save_sota(out_dir: Path, name: str = SOTA_JSON, **kwargs) -> None:
    return save_json(path=out_dir / name, **kwargs)


def load_checkpoint(name: str = CHECKPOINT_PT, strict: bool = True, *, out_dir: Path, **kwargs) -> None:
    state_dict = torch.load(out_dir / name, map_location=torch.device('cpu'))

    for name, module in kwargs.items():  # type: str, nn.Module
        logger.info(f'loading {name}.checkpoint from {out_dir / name}')
        missing_keys, unexpected_keys = module.load_state_dict(state_dict=state_dict[name], strict=strict)

        if not strict:
            for missing_key in missing_keys:
                logger.warning(f'{name}.{missing_key} is missing')

            for unexpected_key in unexpected_keys:
                logger.warning(f'{name}.{unexpected_key} is unexpected')


def save_checkpoint(name: str = CHECKPOINT_PT, *, out_dir: Path, **kwargs) -> None:
    logger.info(f'saving checkpoint ({", ".join(kwargs.keys())}) to {out_dir / name}')
    return torch.save({name: module.state_dict() for name, module in kwargs.items()}, f=out_dir / name)


def extract(path: Path) -> Path:
    logger.info(f'extracting files from {path}')

    if path.name.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as fp:
            fp.extractall(path=path.parent)

    elif path.name.endswith('.tar'):
        with tarfile.open(path, 'r') as fp:
            fp.extractall(path=path.parent)

    elif path.name.endswith('.tar.gz') or path.name.endswith('.tgz'):
        with tarfile.open(path, 'r:gz') as fp:
            fp.extractall(path=path.parent)

    elif path.name.endswith('.tar.bz2') or path.name.endswith('.tbz'):
        with tarfile.open(path, 'r:bz2') as fp:
            fp.extractall(path=path.parent)

    elif path.name.endswith('.gz'):
        with gzip.open(path, mode='rb') as fs:
            with path.with_suffix('').open(mode='wb') as fd:
                shutil.copyfileobj(fs, fd)

    return path


class DownloadMixin(object):
    name: str

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        raise NotImplementedError

    @classmethod
    def paths(cls, root: Path = data_dir, **kwargs) -> List[Path]:
        dataset_name = getattr(cls, 'name', cls.__name__).lower()

        dm = DownloadManager(
            dataset_name=dataset_name,
            download_config=DownloadConfig(
                cache_dir=root / dataset_name,
                extract_compressed_file=True,
                force_extract=True,
                delete_extracted=True,
                download_desc=f'Downloading {dataset_name}',
            ),
        )

        out = []
        for url, *names in cls.urls(**kwargs):
            archive = Path(dm.download_and_extract(url))

            if len(names) == 0:
                out.append(archive)
            else:
                for name in names:
                    out.append(archive / name)

        return out
