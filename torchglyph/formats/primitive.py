from typing import Type, Any

__all__ = [
    'loads_type', 'loads_bool',
    'dumps_type', 'dumps_bool',
]


def loads_bool(string: str) -> bool:
    string = string.strip().lower()
    if string in ('1', 'y', 'yes', 't', 'true'):
        return True
    if string in ('0', 'n', 'no', 'f', 'false'):
        return False
    raise TypeError(f'{string} is not a boolean value')


def loads_type(string: str, *, tp: Type) -> Any:
    if tp is bool:
        return loads_bool(string)
    return tp(string)


def dumps_bool(obj: bool) -> str:
    return f'{obj}'


def dumps_type(obj: Any) -> str:
    if isinstance(obj, bool):
        return dumps_bool(obj)
    return f'{obj}'
