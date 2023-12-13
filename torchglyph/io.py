import logging
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple

from datasets.config import DATASETDICT_JSON_FILENAME, DATASET_INFO_FILENAME
from datasets.download import DownloadConfig, DownloadManager
from datasets.fingerprint import Hasher
from filelock import FileLock

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
