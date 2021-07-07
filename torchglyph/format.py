import uuid
from pathlib import Path

from typing import NamedTuple, List, Iterable, get_type_hints

__all__ = [
    'register_type', 'type_get',
    'load_conll',
]

registry = {}


def register_type(tp):
    registry[get_type_hints(tp)['return']] = tp
    return tp


def type_get(tp):
    return registry.get(tp, tp)


@register_type
def parse_bool(string: str) -> bool:
    string = string.strip().lower()
    if string in ('1', 'y', 'yes', 't', 'true'):
        return True
    if string in ('0', 'n', 'no', 'f', 'false'):
        return False
    raise TypeError(f'{string} is not {bool}')


def load_conll(path: Path, config: NamedTuple, sep: str = ' ',
               blank: str = '', encoding: str = 'utf-8') -> Iterable[NamedTuple]:
    Token = NamedTuple(
        f'Token_{str(uuid.uuid4())[:8]}',
        [(name, List[config._field_types[name]])
         for name in config._fields if not name.endswith('_')],
    )

    with path.open(mode='r', encoding=encoding) as fp:
        tokens = []
        for raw in fp:
            raw = raw.strip()
            if raw != blank:
                tokens.append([
                    type_get(config._field_types[name])(data)
                    for data, name in zip(raw.split(sep=sep), config._fields)
                    if not name.endswith('_')
                ])
            elif len(tokens) != 0:
                yield Token(*zip(*tokens))
                tokens = []

        if len(tokens) != 0:
            yield Token(*zip(*tokens))
