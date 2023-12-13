from contextlib import contextmanager
from pathlib import Path

from datasets.config import DATASETDICT_JSON_FILENAME
from datasets.fingerprint import Hasher
from filelock import FileLock

from torchglyph import DEBUG


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
    uuid2 = get_hash(**kwargs, __torchglyph=DEBUG)

    if DEBUG and not (out_dir / uuid1).exists():
        return out_dir / uuid2

    return out_dir / uuid1


def all_exists(out_dir: Path, *names: str) -> bool:
    return all(
        (out_dir / name).exists()
        for name in (DATASETDICT_JSON_FILENAME, *names)
    )
