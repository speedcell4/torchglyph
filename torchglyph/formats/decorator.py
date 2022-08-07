import inspect
import json
from logging import getLogger
from pathlib import Path
from typing import Union, Any

import torch
from filelock import FileLock

logger = getLogger(__name__)


def with_lock(fn_or_method):
    def fn(path: Union[str, Path], *args, **kwargs):
        lock_file = Path(path).with_name(f'{path.name}.lock').resolve()

        with FileLock(lock_file=str(lock_file)):
            return fn_or_method(path, *args, **kwargs)

    def method(self, path: Union[str, Path], *args, **kwargs):
        lock_file = Path(path).with_name(f'{path.name}.lock').resolve()

        with FileLock(lock_file=str(lock_file)):
            return fn_or_method(self, path, *args, **kwargs)

    return method if inspect.ismethod(fn_or_method) else fn


def cache_as(suffix: str, load_fn, save_fn):
    def _cache_as(fn_or_method):
        def fn(path: Union[str, Path], *args, **kwargs):
            cache = Path(path).with_name(f'{path.name}.{suffix}').resolve()

            if cache.exists():
                logger.info(f'loading from {cache}')
                return load_fn(path=cache)
            else:

                logger.info(f'saving to {cache}')
                obj = fn_or_method(path, *args, kwargs)
                return save_fn(path=cache, obj=obj)

        def method(self, path: Union[str, Path], *args, **kwargs):
            cache = Path(path).with_name(f'{path.name}.{suffix}').resolve()

            if cache.exists():
                logger.info(f'loading from {cache}')
                return load_fn(path=cache)
            else:

                logger.info(f'saving to {cache}')
                return save_fn(path=cache, obj=fn_or_method(self, path, *args, kwargs))

        return method if inspect.ismethod(fn_or_method) else fn

    return _cache_as


def load_json(path: Path) -> Any:
    with path.open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


def save_json(path: Path, obj: Any) -> None:
    with path.open(mode='w', encoding='utf-8') as fp:
        json.dump(fp=fp, obj=obj, indent=2, ensure_ascii=False)


def load_pt(path: Path) -> Any:
    return torch.load(f=str(path.resolve()), map_location=torch.device('cpu'))


def save_pt(path: Path, obj: Any) -> None:
    torch.save(f=str(path.resolve()), obj=obj)


cache_as_pt = cache_as('pt', load_fn=load_pt, save_fn=save_pt)
cache_as_json = cache_as('json', load_fn=load_json, save_fn=save_json)
