import os
import random
import socket
import warnings
from contextlib import contextmanager
from datetime import datetime
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
from datasets.config import DATASETDICT_JSON_FILENAME
from datasets.fingerprint import Hasher
from filelock import FileLock
from torch import distributed

from torchglyph import DEBUG
from torchglyph.dist import get_device
from torchglyph.logger import init_logger
from torchglyph.serde import save_args

logger = getLogger(__name__)


@contextmanager
def lock_dir(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(out_dir.resolve() / '.lock')):
        yield


def get_hash(**kwargs) -> str:
    hasher = Hasher()

    for key, value in sorted(kwargs.items()):
        hasher.update(key)
        hasher.update(value)

    return hasher.hexdigest()


def get_cache(out_dir: Path, **kwargs) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    uuid1 = get_hash(**kwargs)
    uuid2 = get_hash(**kwargs, __debug=DEBUG)

    if DEBUG and not (out_dir / uuid1).exists():
        return out_dir / uuid2

    return out_dir / uuid1


def all_exists(out_dir: Path, *names: str) -> bool:
    return all(
        (out_dir / name).exists()
        for name in (DATASETDICT_JSON_FILENAME, *names)
    )


def init_seed(rank: int, seed: int) -> None:
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return logger.warning(f'{rank}.seed <- {seed}')


def init_process_group(rank: int, port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'
    distributed.init_process_group(
        backend='nccl', init_method=f'env://',
        world_size=torch.cuda.device_count(), rank=rank,
    )

    torch.cuda.set_device(rank)
    torch.cuda.synchronize(rank)


def get_port() -> int:
    sock = socket.socket()
    sock.bind(('', 0))

    _, port = sock.getsockname()
    return port


def init_rank(rank: int, port: int, out_dir: Path, /, seed: int = 42):
    if rank == -1:
        init_logger(rank=0, out_dir=out_dir)
        init_seed(rank=0, seed=seed)
    else:
        init_process_group(rank=rank, port=port)
        init_logger(rank=rank, out_dir=out_dir)
        init_seed(rank=rank, seed=seed)

    return get_device()


def init_study(study: str, *, project_out_dir: Path, **kwargs) -> Path:
    with lock_dir(out_dir=project_out_dir / study):
        uuid = get_hash(__datetime=datetime.now() if DEBUG or study == 'demo' else None, **kwargs)
        out_dir = project_out_dir / study / uuid

        if out_dir.exists():
            warnings.warn('duplicated experiment')
            exit()

        out_dir.mkdir(parents=True, exist_ok=False)
        save_args(out_dir=out_dir, hostname=socket.gethostname(), **kwargs)

    return out_dir
