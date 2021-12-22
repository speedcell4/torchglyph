import json
from logging import getLogger
from pathlib import Path
from typing import Any, Set, Union
from typing import List, Tuple, Dict

import torch
from tabulate import tabulate
from torch import nn

logger = getLogger(__name__)

LOG_TXT = 'log.txt'
ARGS_JSON = 'args.json'
SOTA_JSON = 'sota.json'
CHECKPOINT_PT = 'checkpoint.pt'


def save_kwargs(name: str, *, out_dir: Path, **kwargs) -> None:
    if (out_dir / name).exists():
        with (out_dir / name).open(mode='r', encoding='utf-8') as fp:
            kwargs = {**json.load(fp=fp), **kwargs}

    with (out_dir / name).open(mode='w', encoding='utf-8') as fp:
        json.dump(kwargs, fp=fp, indent=2, sort_keys=True)


def save_args(*, out_dir: Path, **kwargs) -> None:
    return save_kwargs(name=ARGS_JSON, out_dir=out_dir, **kwargs)


def save_sota(*, out_dir: Path, **kwargs) -> None:
    return save_kwargs(name=SOTA_JSON, out_dir=out_dir, **kwargs)


def save_checkpoint(name: str = CHECKPOINT_PT, *, out_dir: Path, **kwargs) -> None:
    logger.info(f'saving checkpoint ({", ".join(kwargs.keys())}) to {out_dir / name}')
    return torch.save({name: module.state_dict() for name, module in kwargs.items()}, f=out_dir / name)


def load_checkpoint(name: str = CHECKPOINT_PT, strict: bool = True, *, out_dir: Path, **kwargs) -> None:
    state_dict = torch.load(out_dir / name, map_location=torch.device('cpu'))

    for name, module in kwargs.items():  # type: str, nn.Module
        logger.info(f'loading {name}.checkpoint from {out_dir / name}')
        missing_keys, unexpected_keys = module.load_state_dict(state_dict=state_dict[name], strict=strict)

        if not strict:
            for missing_key in missing_keys:
                logger.warning(f'{name}.{missing_key} is missing')

            for unexpected_key in unexpected_keys:
                logger.warning(f'{name}.{unexpected_key} is unexpected')


def fetch(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with (out_dir / SOTA_JSON).open(mode='r', encoding='utf-8') as fp:
        sota = json.load(fp)
    with (out_dir / ARGS_JSON).open(mode='r', encoding='utf-8') as fp:
        args = json.load(fp)

    return sota, {**args, 'path': out_dir / LOG_TXT}


def group_keys(keys: Set[str], args: List[Dict[str, Any]]):
    has_path = False
    major, common = [], {}

    for key in keys:
        values = set(a.get(key, '-') for a in args)
        if len(values) > 1:
            if key == 'path':
                has_path = True
            else:
                major.append(key)
        else:
            common[key] = args[0][key]

    major = sorted(major)
    if has_path:
        major = major + ['path']

    return major, list(common.items())


def reduce_metric(data: List[float], expand: bool) -> Union[Tuple[str], Tuple[str, str, int]]:
    tensor = torch.tensor(data, dtype=torch.float32)
    if expand:
        return f'{tensor.mean().item():.4f}',
    std, mean = torch.std_mean(tensor, unbiased=True)
    return f'{mean.item():.4f}', f'{std.item():.2f}', len(data)


def summary(path: List[Path], metric: str, sort: bool = True,
            common: bool = False, expand: bool = False, fmt: str = 'pretty'):
    sota, args = zip(*[
        fetch(out_dir=out_dir)
        for p in path for out_dir in p.iterdir()
        if out_dir.is_dir() and (out_dir / ARGS_JSON).exists() and (out_dir / SOTA_JSON).exists()
    ])

    if expand:
        ignores = ('device', 'seed')
    else:
        ignores = ('device', 'seed', 'path')

    keys = set(k for a in args for k in a.keys() if k not in ignores)
    keys, tabular_data = group_keys(keys=keys, args=args)

    if common:
        print(tabulate(tabular_data=tabular_data, headers=['key', 'value'], tablefmt=fmt))
    else:
        tabular_data = {}
        for s, a in zip(sota, args):
            if metric in s:
                vs = tuple(a.get(key, '-') for key in keys)
                tabular_data.setdefault(vs, []).append(s[metric])

        tabular_data = [
            [*reduce_metric(data, expand), *vs]
            for vs, data in tabular_data.items()
        ]
        if sort:
            tabular_data = list(sorted(tabular_data, key=lambda item: item[0], reverse=False))

        if expand:
            headers = [metric, *keys]
        else:
            headers = [metric, 'std', '*', *keys]

        print(tabulate(tabular_data=tabular_data, headers=headers, tablefmt=fmt))
