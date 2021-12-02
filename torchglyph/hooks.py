import json
from pathlib import Path
from typing import Tuple, Dict, Any, List

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


def fetch(out_dir: Path, ignores: Tuple[str, ...]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with (out_dir / SOTA_JSON).open(mode='r', encoding='utf-8') as fp:
        sota = json.load(fp)
        sota = {key: value for key, value in sota.items() if key not in ignores}

    with (out_dir / ARGS_JSON).open(mode='r', encoding='utf-8') as fp:
        args = {**json.load(fp), 'path': out_dir / LOG_TXT}
        args = {key: value for key, value in args.items() if key not in ignores}

    return sota, args


def summary(path: List[Path], keys: Tuple[str, ...], ignores: Tuple[str, ...] = ('device', 'seed', 'path')):
    sota, args = zip(*[
        fetch(out_dir=out_dir, ignores=ignores)
        for p in path for out_dir in p.iterdir()
        if out_dir.is_dir() and (out_dir / ARGS_JSON).exists() and (out_dir / SOTA_JSON).exists()
    ])
