from typing import Type, Any

__all__ = [
    'loads_type', 'loads_bool',
    'dumps_type', 'dumps_bool',
]


def loads_bool(s: str) -> bool:
    s = s.strip().lower()
    if s in ('1', 'y', 'yes', 't', 'true'):
        return True
    if s in ('0', 'n', 'no', 'f', 'false'):
        return False
    raise TypeError(f'{s} is not a boolean value')


def loads_type(s: str, *, tp: Type) -> Any:
    if tp is bool:
        return loads_bool(s)
    return tp(s)


def dumps_bool(v: bool) -> str:
    return f'{v}'


def dumps_type(v: Any) -> str:
    if isinstance(v, bool):
        return dumps_bool(v)
    return f'{v}'
