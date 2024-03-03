import os
import random
import socket
import warnings
from datetime import datetime
from pathlib import Path
from typing import Type, Union

import numpy as np
import torch
from filelock import FileLock
from torch import distributed

from torchglyph import DEBUG
from torchglyph.dist import get_device, logger
from torchglyph.logger import init_logger
from torchglyph.serde import get_cache, save_args


def init_dir(study: str, *, project_out_dir: Path, **kwargs) -> Path:
    with FileLock(project_out_dir / study / '.lock'):
        try:
            out_dir = get_cache(
                project_out_dir / study, exist_ok=False, **kwargs,
                __ts=datetime.now() if DEBUG or study == 'demo' else None,
            )
        except FileExistsError:
            warnings.warn('duplicated experiment')
            exit()

        save_args(out_dir=out_dir, hostname=socket.gethostname(), **kwargs)

    return out_dir


def get_port() -> int:
    sock = socket.socket()
    sock.bind(('', 0))

    _, port = sock.getsockname()
    return port


def init_process_group(*, rank: int, port: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{port}'
    distributed.init_process_group(
        backend='nccl', init_method=f'env://',
        world_size=torch.cuda.device_count(), rank=rank,
    )

    torch.cuda.set_device(rank)
    torch.cuda.synchronize(rank)


def init_seed(seed: int = 42, *, rank: int) -> None:
    seed = seed + rank

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return logger.warning(f'#{rank}.seed <- {seed}')


def init_rank(rank: int, out_dir: Path, port: int, /,
              setup_process_group: Union[Type[init_process_group]] = init_process_group,
              setup_logger: Union[Type[init_logger]] = init_logger,
              setup_seed: Union[Type[init_seed]] = init_seed):
    if rank == -1:
        setup_logger(rank=0, out_dir=out_dir)
        setup_seed(rank=0)
    else:
        setup_process_group(rank=rank, port=port)
        setup_logger(rank=rank, out_dir=out_dir)
        setup_seed(rank=rank)

    return get_device()
