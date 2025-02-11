import functools
import json
from pathlib import Path
from typing import Any, Union

from torch import distributed


def get_rank() -> int:
    if distributed.is_initialized():
        return distributed.get_rank()

    return 0


def is_master_process() -> bool:
    if distributed.is_initialized():
        return distributed.get_rank() == 0

    return True


def master_only(fn):
    @functools.wraps(fn)
    def _fn(*args, **kwargs):
        if is_master_process():
            return fn(*args, **kwargs)

    return _fn


def load_json(path: Union[Path, str]) -> Any:
    with Path(path).open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


load_json_master_only = master_only(load_json)


def dump_json(path: Union[Path, str], obj: Any) -> None:
    with Path(path).open(mode='w', encoding='utf-8') as fp:
        json.dump(obj, fp=fp, indent=2, ensure_ascii=False)


dump_json_master_only = master_only(dump_json)
