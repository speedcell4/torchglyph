import json
from logging import getLogger
from pathlib import Path
from typing import Any, Set
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


def fetch_one(out_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with (out_dir / SOTA_JSON).open(mode='r', encoding='utf-8') as fp:
        sota = json.load(fp)
    with (out_dir / ARGS_JSON).open(mode='r', encoding='utf-8') as fp:
        args = json.load(fp)

    return {**args, 'path': out_dir / LOG_TXT}, sota


def fetch_all(paths: List[Path]):
    return zip(*[
        fetch_one(out_dir=out_dir)
        for p in paths for out_dir in p.iterdir()
        if out_dir.is_dir() and (out_dir / ARGS_JSON).exists() and (out_dir / SOTA_JSON).exists()
    ])


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


def reduce_metric(metrics: List[Tuple[float, ...]], expand: bool):
    metrics = torch.tensor(metrics, dtype=torch.float32)
    *mean, epoch1, epoch2 = [round(m, 4) for m in metrics.mean(dim=0).detach().tolist()]
    epoch1 = int(epoch1)
    epoch2 = int(epoch2)
    std = round(metrics[:, -3].std(unbiased=True).item(), 4)
    if expand:
        return *mean, epoch1, epoch2
    return *mean, std, len(metrics), epoch1, epoch2


def summary(path: List[Path], metrics: Tuple[str, ...], sort: bool = True,
            common: bool = False, expand: bool = False, fmt: str = 'pretty'):
    args, sota = fetch_all(path)

    ignores = ('study', 'device', 'seed')
    if not expand:
        ignores = (*ignores, 'path')

    keys = set(k for a in args for k in a.keys() if k not in ignores)
    keys, tabular_data = group_keys(keys=keys, args=args)

    if common:
        print(tabulate(tabular_data=tabular_data, headers=['key', 'value'], tablefmt=fmt))
    else:
        if metrics[-1].startswith('dev_'):
            epoch = 'dev_epoch'
        elif metrics[-1].startswith('test_'):
            epoch = 'test_epoch'
        else:
            epoch = 'epoch'

        tabular_data = {}
        for s, a in zip(sota, args):
            if all(m in s for m in metrics):
                vs = tuple(a.get(k, '-') for k in keys)
                ms = tuple(s[m] for m in metrics)
                tabular_data.setdefault(vs, []).append((*ms, s[epoch], s['epoch']))

        tabular_data = [
            [*reduce_metric(metrics, expand), *vs]
            for vs, metrics in tabular_data.items()
        ]
        if sort:
            tabular_data = list(sorted(tabular_data, key=lambda item: item[len(metrics) - 1], reverse=False))

        if expand:
            headers = [*metrics, 's', 'e', *keys]
        else:
            headers = [*metrics, 'std', '*', 's', 'e', *keys]

        print(tabulate(tabular_data=tabular_data, headers=headers, tablefmt=fmt))
