import socket
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from datasets.config import DATASETDICT_JSON_FILENAME
from datasets.fingerprint import Hasher
from filelock import FileLock

from torchglyph import DEBUG
from torchglyph.serde import save_args


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


def init_dir(study: str, *, project_out_dir: Path, **kwargs) -> Path:
    with lock_dir(out_dir=project_out_dir / study):
        uuid = get_hash(__datetime=datetime.now() if DEBUG or study == 'demo' else None, **kwargs)
        out_dir = project_out_dir / study / uuid

        if out_dir.exists():
            warnings.warn('duplicated experiment')
            exit()

        out_dir.mkdir(parents=True, exist_ok=False)
        save_args(out_dir=out_dir, hostname=socket.gethostname(), **kwargs)

    return out_dir
