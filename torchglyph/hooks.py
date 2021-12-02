import json
from pathlib import Path

ARGS_JSON = 'args.json'
SOTA_JSON = 'sota.json'


def save_kwargs(name: str, *, out_dir: Path, **kwargs) -> None:
    if (out_dir / name).exists():
        with (out_dir / name).open(mode='r', encoding='utf-8') as fp:
            kwargs = {**kwargs, **json.load(fp=fp)}

    with (out_dir / name).open(mode='w', encoding='utf-8') as fp:
        json.dump(kwargs, fp=fp, indent=2, sort_keys=True)


def save_args(name: str = ARGS_JSON, *, out_dir: Path, **kwargs) -> None:
    return save_kwargs(name=name, out_dir=out_dir, **kwargs)


def save_sota(name: str = SOTA_JSON, *, out_dir: Path, **kwargs) -> None:
    return save_kwargs(name=name, out_dir=out_dir, **kwargs)
