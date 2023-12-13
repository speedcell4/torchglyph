import json
from logging import getLogger
from pathlib import Path
from typing import Any

import yaml

logger = getLogger(__name__)


def load_json(path: Path) -> Any:
    with path.open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


def load_yaml(path: Path) -> Any:
    with path.open(mode='r', encoding='utf-8') as fp:
        return yaml.load(stream=fp, Loader=yaml.CLoader)


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
    return save_json(path=load_json(out_dir / name), **kwargs)


def save_sota(out_dir: Path, name: str = SOTA_FILENAME, **kwargs) -> None:
    return save_json(path=load_json(out_dir / name), **kwargs)
