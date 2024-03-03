import socket
import warnings
from datetime import datetime
from pathlib import Path

from filelock import FileLock

from torchglyph import DEBUG
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
