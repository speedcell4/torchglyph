import json
from logging import getLogger
from pathlib import Path
from typing import Any, Set, List, Tuple, Dict, Iterable, Type

import torch
from filelock import FileLock
from tabulate import tabulate
from torch import nn

logger = getLogger(__name__)

LOG_TXT = 'log.txt'
ARGS_JSON = 'args.json'
SOTA_JSON = 'sota.json'
CHECKPOINT_PT = 'checkpoint.pt'


def load_kwargs(*, out_dir: Path, _kwargs_name: str):
    with FileLock(str(out_dir / f'{_kwargs_name}.lock')):
        if (out_dir / _kwargs_name).exists():
            with (out_dir / _kwargs_name).open(mode='r', encoding='utf-8') as fp:
                return json.load(fp=fp)


def load_args(*, out_dir: Path):
    return load_kwargs(out_dir=out_dir, _kwargs_name=ARGS_JSON)


def load_sota(*, out_dir: Path):
    return load_kwargs(out_dir=out_dir, _kwargs_name=SOTA_JSON)


def save_kwargs(*, out_dir: Path, _kwargs_name: str, **kwargs) -> None:
    with FileLock(str(out_dir / f'{_kwargs_name}.lock')):
        if (out_dir / _kwargs_name).exists():
            with (out_dir / _kwargs_name).open(mode='r', encoding='utf-8') as fp:
                kwargs = {**json.load(fp=fp), **kwargs}

        with (out_dir / _kwargs_name).open(mode='w', encoding='utf-8') as fp:
            json.dump(kwargs, fp=fp, indent=2, sort_keys=True, ensure_ascii=False)


def save_args(*, out_dir: Path, **kwargs) -> None:
    return save_kwargs(out_dir=out_dir, _kwargs_name=ARGS_JSON, **kwargs)


def save_sota(*, out_dir: Path, **kwargs) -> None:
    return save_kwargs(out_dir=out_dir, _kwargs_name=SOTA_JSON, **kwargs)


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


def save_checkpoint(name: str = CHECKPOINT_PT, *, out_dir: Path, **kwargs) -> None:
    logger.info(f'saving checkpoint ({", ".join(kwargs.keys())}) to {out_dir / name}')
    return torch.save({name: module.state_dict() for name, module in kwargs.items()}, f=out_dir / name)


def fetch_one(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with (out_dir / SOTA_JSON).open(mode='r', encoding='utf-8') as fp:
        sota = json.load(fp)
    with (out_dir / ARGS_JSON).open(mode='r', encoding='utf-8') as fp:
        args = json.load(fp)

    return {**args, 'path': out_dir / LOG_TXT}, sota


def recur_dir(path: Path) -> Iterable[Path]:
    for out_dir in path.iterdir():
        if out_dir.is_dir():
            if (out_dir / ARGS_JSON).exists() and (out_dir / SOTA_JSON).exists():
                yield out_dir
            else:
                yield from recur_dir(out_dir)


def fetch_all(paths: List[Path]):
    return zip(*[
        fetch_one(out_dir=out_dir)
        for p in paths for out_dir in recur_dir(p)
    ])


def frozen(item: Any) -> Any:
    if isinstance(item, list):
        return tuple(item)
    if isinstance(item, set):
        return frozenset(item)
    return item


def group_keys(keys: Set[str], args: List[Dict[str, Any]]):
    has_path = False
    major, common = [], {}

    for key in keys:
        values = set(frozen(a.get(key, '-')) for a in args)
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


def reduce_metric(metrics: List[Tuple[float, ...]]):
    metrics = torch.tensor(metrics, dtype=torch.float32)
    return [round(m, 2) for m in metrics.mean(dim=0).detach().tolist()]


def group_data(group: Tuple[str, ...] = (), *, data, headers, metrics):
    if len(group) == 0:
        return data

    group_indices = [headers.index(key) for key in group]
    metric_indices = [headers.index(key) for key in metrics[::-1]]

    groups = {}
    for datum in data:
        sort_key = tuple(datum[index] for index in group_indices)
        groups.setdefault(sort_key, []).append(datum)

    return [
        max(values, key=lambda datum: [datum[index] for index in metric_indices])
        for values in groups.values()
    ]


def sort_data(sort: Tuple[str, ...] = (), *, data, headers, metrics):
    if len(sort) == 0:
        sort = metrics[::-1]

    indices = [headers.index(key) for key in sort]
    return list(sorted(data, key=lambda datum: [datum[index] for index in indices]))


def summary(path: List[Path], metrics: Tuple[str, ...],
            group: Type[group_data] = group_data, sort: Type[sort_data] = sort_data,
            ignore: Tuple[str, ...] = ('study', 'device', 'seed', 'hostname'),
            common: bool = False, expand: bool = False, fmt: str = 'pretty'):
    args, sota = fetch_all(path)

    if not expand:
        ignore = (*ignore, 'path')

    keys = set(k for a in args for k in a.keys() if k not in ignore)
    keys, data = group_keys(keys=keys, args=args)

    if common:
        print(tabulate(tabular_data=data, headers=['key', 'value'], tablefmt=fmt))
    else:
        data = {}
        for s, a in zip(sota, args):
            if all(m in s for m in metrics):
                vs = tuple(frozen(a.get(k, '-')) for k in keys)
                ms = tuple(s[m] for m in metrics)
                data.setdefault(vs, []).append(ms)

        data = [
            [*reduce_metric(metrics), len(metrics), *vs]
            for vs, metrics in data.items()
        ]
        headers = [*metrics, '@', *keys]

        data = group(data=data, headers=headers, metrics=metrics)
        data = sort(data=data, headers=headers, metrics=metrics)

        print(tabulate(tabular_data=data, headers=headers, tablefmt=fmt))
