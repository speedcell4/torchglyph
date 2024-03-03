import json
from logging import getLogger
from pathlib import Path
from typing import Any

import yaml
from datasets.config import DATASETDICT_JSON_FILENAME, DATASET_INFO_FILENAME
from datasets.fingerprint import Hasher

from torchglyph import DEBUG

logger = getLogger(__name__)


def get_hash(**kwargs) -> str:
    hasher = Hasher()

    for key, value in sorted(kwargs.items()):
        hasher.update(key)
        hasher.update(value)

    return hasher.hexdigest()


def get_cache(path: Path, exist_ok: bool = True, **kwargs) -> Path:
    cache = path / get_hash(**kwargs, __torchglyph=DEBUG)
    cache.mkdir(parents=True, exist_ok=exist_ok)
    return cache


def is_cached(path: Path, *names: str) -> bool:
    return all([
        all((path / name).exists() for name in names),
        any((path / name).exists() for name in (DATASET_INFO_FILENAME, DATASETDICT_JSON_FILENAME)),
    ])


def load_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open(mode='r', encoding='utf-8') as fp:
            return json.load(fp=fp)
    except FileNotFoundError as error:
        if default is not None:
            return default
        raise error


def load_yaml(path: Path, default: Any = None) -> Any:
    try:
        with path.open(mode='r', encoding='utf-8') as fp:
            return yaml.load(stream=fp, Loader=yaml.CLoader)
    except FileNotFoundError as error:
        if default is not None:
            return default
        raise error


def save_json(path: Path, **kwargs) -> None:
    if not path.exists():
        logger.info(f'saving to {path}')
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode='w', encoding='utf-8') as fp:
        return json.dump(obj=kwargs, fp=fp, indent=2, ensure_ascii=False)


def save_yaml(path: Path, **kwargs) -> None:
    if not path.exists():
        logger.info(f'saving to {path}')
        path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(mode='w', encoding='utf-8') as fp:
        return yaml.dump(data=kwargs, stream=fp, indent=2, allow_unicode=True)


ARGS_FILENAME = 'args.json'
SOTA_FILENAME = 'sota.json'


def load_args(out_dir: Path, name: str = ARGS_FILENAME) -> Any:
    return load_json(path=out_dir / name)


def load_sota(out_dir: Path, name: str = SOTA_FILENAME) -> Any:
    return load_json(path=out_dir / name)


def save_args(out_dir: Path, name: str = ARGS_FILENAME, **kwargs) -> None:
    return save_json(path=out_dir / name, **{**load_json(out_dir / name, default={}), **kwargs})


def save_sota(out_dir: Path, name: str = SOTA_FILENAME, **kwargs) -> None:
    return save_json(path=out_dir / name, **{**load_json(out_dir / name, default={}), **kwargs})
