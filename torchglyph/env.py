import socket
import warnings
from datetime import datetime
from pathlib import Path
from typing import Type, Union

from torchglyph import DEBUG
from torchglyph.dist import get_device, init_process, init_seed
from torchglyph.io import hf_hash, lock_folder
from torchglyph.logger import init_logger
from torchglyph.serde import save_args


def timestamp(*, time_format: str = r'%y%m%d-%H%M%S') -> str:
    return datetime.strftime(datetime.now(), time_format).strip()


def init_env(study: str, *, project_out_dir: Path, **kwargs) -> Path:
    with lock_folder(path=project_out_dir / study):
        uuid = hf_hash(**kwargs, __ts=timestamp() if DEBUG or study == 'demo' else None)
        out_dir = project_out_dir / study / uuid

        if out_dir.exists():
            warnings.warn('duplicated experiment')
            exit()

        out_dir.mkdir(parents=True, exist_ok=False)
        save_args(out_dir=out_dir, hostname=socket.gethostname(), **kwargs)

    return out_dir


def init_rank(rank: int, out_dir: Path, /,
              setup_process: Union[Type[init_process]] = init_process,
              setup_logger: Union[Type[init_logger]] = init_logger,
              setup_seed: Union[Type[init_seed]] = init_seed):
    if rank == -1:
        setup_logger(rank=0, out_dir=out_dir)
        setup_seed(rank=0)
    else:
        setup_process(rank=rank, port=8888)
        setup_logger(rank=rank, out_dir=out_dir)
        setup_seed(rank=rank)

    return get_device()
