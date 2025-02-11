import functools
import json
import random
import socket
import warnings
from logging import getLogger
from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch
from datasets.fingerprint import Hasher
from filelock import FileLock
from torch import Tensor, distributed

SOTA_FILENAME = 'sota.json'
ARGS_FILENAME = 'args.json'

logger = getLogger(__name__)


def get_rank() -> int:
    if distributed.is_initialized():
        return distributed.get_rank()

    return 0


def get_local_rank() -> int:
    if distributed.is_initialized():
        return distributed.get_node_local_rank()

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


def all_gather(tensor: Tensor, world_size: int = None) -> List[Tensor]:
    if distributed.is_initialized():
        if world_size is None:
            world_size = distributed.get_world_size()

        tensor_list = [
            tensor.new_empty(size)
            for size in all_gather_object(tuple(tensor.size()), world_size=world_size)
        ]
        distributed.all_gather(tensor_list=tensor_list, tensor=tensor)
        return tensor_list

    return [tensor]


def load_json(path: Union[Path, str]) -> Any:
    with Path(path).open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


def save_json(path: Union[Path, str], obj: Any) -> None:
    with Path(path).open(mode='w', encoding='utf-8') as fp:
        json.dump(obj, fp=fp, indent=2, ensure_ascii=False)


@master_only
def save_args(obj: Any, *, out_dir: Path) -> None:
    return save_json(path=out_dir / ARGS_FILENAME, obj=obj)


@master_only
def save_sota(obj: Any, *, out_dir: Path) -> None:
    return save_json(path=out_dir / SOTA_FILENAME, obj=obj)


def hash_kwargs(**kwargs) -> str:
    hasher = Hasher()

    for key, value in sorted(kwargs.items()):
        hasher.update(key)
        hasher.update(value)

    return hasher.hexdigest()


def init_dir(study: str, *, project_out_dir: Path, **kwargs) -> Path:
    out_dir = project_out_dir / study / hash_kwargs(**kwargs)

    if is_master_process():
        with FileLock(project_out_dir / '.lock'):
            try:
                out_dir.parent.mkdir(parents=True, exist_ok=study == 'demo')
                save_args(obj=kwargs, out_dir=out_dir)
            except FileExistsError:
                warnings.warn('duplicated experiment')
                exit()

    if distributed.is_initialized():
        distributed.barrier()

    return out_dir


def init_seed(seed: int = 42) -> None:
    rank = get_rank()
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    if distributed.is_initialized():
        distributed.barrier()

    return logger.warning(f'#{rank} ({socket.gethostname()}-{get_local_rank()}) <- {seed}')
