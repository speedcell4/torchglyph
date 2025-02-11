import functools
import json
from pathlib import Path
from typing import Any, List, Union

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


def all_gather_object(obj: Any, world_size: int = None) -> List[Any]:
    if distributed.is_initialized():
        if world_size is None:
            world_size = distributed.get_world_size()

        object_list = [None for _ in range(world_size)]
        distributed.all_gather_object(object_list=object_list, obj=obj)
        return object_list

    return [obj]


def load_json(path: Union[Path, str]) -> Any:
    with Path(path).open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


load_json_master_only = master_only(load_json)


def dump_json(path: Union[Path, str], obj: Any) -> None:
    with Path(path).open(mode='w', encoding='utf-8') as fp:
        json.dump(obj, fp=fp, indent=2, ensure_ascii=False)


dump_json_master_only = master_only(dump_json)
