import json
from pathlib import Path
from typing import Tuple, Dict, Any

LOG_TXT = 'log.txt'
ARGS_JSON = 'args.json'
SOTA_JSON = 'sota.json'


def save_kwargs(name: str, *, out_dir: Path, **kwargs) -> None:
    if (out_dir / name).exists():
        with (out_dir / name).open(mode='r', encoding='utf-8') as fp:
            kwargs = {**kwargs, **json.load(fp=fp)}

    with (out_dir / name).open(mode='w', encoding='utf-8') as fp:
        json.dump(kwargs, fp=fp, indent=2, sort_keys=True)


def save_args(*, out_dir: Path, **kwargs) -> None:
    return save_kwargs(name=ARGS_JSON, out_dir=out_dir, **kwargs)


def save_sota(*, out_dir: Path, **kwargs) -> None:
    return save_kwargs(name=SOTA_JSON, out_dir=out_dir, **kwargs)


def fetch(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with (out_dir / ARGS_JSON).open(mode='r', encoding='utf-8') as fp:
        args = json.load(fp)

    with (out_dir / SOTA_JSON).open(mode='r', encoding='utf-8') as fp:
        sota = json.load(fp)

    return sota, {**args, 'path': out_dir / LOG_TXT}
