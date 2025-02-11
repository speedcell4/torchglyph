import json
from pathlib import Path
from typing import Any, Union


def load_json(path: Union[Path, str]) -> Any:
    with Path(path).open(mode='r', encoding='utf-8') as fp:
        return json.load(fp=fp)


def dump_json(path: Union[Path, str], obj: Any) -> None:
    with Path(path).open(mode='w', encoding='utf-8') as fp:
        json.dump(obj, fp=fp, indent=2, ensure_ascii=False)
