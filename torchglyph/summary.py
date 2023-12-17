from pathlib import Path
from typing import List, Tuple

from pandas import DataFrame

from torchglyph.logger import LOG_FILENAME
from torchglyph.serde import load_args, load_sota


def load_study(out_dir: Path):
    args = load_args(out_dir=out_dir)
    sota = load_sota(out_dir=out_dir)

    return {**args, 'path': out_dir / LOG_FILENAME}, sota


def load_studies(out_dir: Path) -> Tuple[List[str]]:
    data, keys = [], set()

    for path in out_dir.iterdir():
        try:
            args, sota = load_study(out_dir=path)
            data.append({**args, **sota})
            keys.add(args.keys())
        except FileNotFoundError:
            pass

    return DataFrame.from_lists(data), list(keys)
