from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple, Type

import torch
from tabulate import tabulate

from torchglyph.io import ARGS_JSON, SOTA_JSON, load_args, load_sota
from torchglyph.logger import LOG_TXT

logger = getLogger(__name__)

SUMMARY_IGNORES = ('study', 'device', 'seed', 'hostname', 'port', 'checkpoint')
SUMMARY_IGNORES = SUMMARY_IGNORES + tuple(f'co-{ignore}' for ignore in SUMMARY_IGNORES)


def iter_dir(path: Path) -> Iterable[Path]:
    for path in path.iterdir():
        if path.is_dir():
            if (path / ARGS_JSON).exists() and (path / SOTA_JSON).exists():
                yield path
            else:
                yield from iter_dir(path)


def fetch_one(out_dir: Path):
    args = load_args(out_dir=out_dir)
    sota = load_sota(out_dir=out_dir)

    return {**args, 'path': out_dir / LOG_TXT}, sota


def fetch_all(paths: List[Path]):
    return zip(*[
        fetch_one(out_dir=out_dir)
        for path in paths for out_dir in iter_dir(path)
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


def margin_data(margin: Tuple[str, ...] = (), *, data, headers, metrics):
    if len(margin) == 0:
        return data

    group_indices = [headers.index(key) for key in margin]
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
            margin: Type[margin_data] = margin_data, sort: Type[sort_data] = sort_data,
            ignore: Tuple[str, ...] = SUMMARY_IGNORES,
            common: bool = False, expand: bool = False, fmt: str = 'pretty'):
    args, sota = fetch_all(path)

    if not expand:
        ignore = (*ignore, 'path')

    keys = set(k for a in args for k in a.keys() if k not in ignore)
    keys, data = group_keys(keys=keys, args=args)

    if common:
        print(tabulate(tabular_data=sorted(data), headers=['key', 'value'], tablefmt=fmt))
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

        data = margin(data=data, headers=headers, metrics=metrics)
        data = sort(data=data, headers=headers, metrics=metrics)

        print(tabulate(tabular_data=data, headers=headers, tablefmt=fmt))
