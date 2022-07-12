from typing import Any

from torchglyph.proc.abc import Proc


class ToStr(Proc):
    def __call__(self, data: Any, **kwargs) -> str:
        return str(data)


class ToInt(Proc):
    def __call__(self, data: Any, **kwargs) -> int:
        return int(data)


class ToBool(Proc):
    def __call__(self, data: Any, **kwargs) -> bool:
        return bool(data)


class ToFloat(Proc):
    def __call__(self, data: Any, **kwargs) -> float:
        return float(data)


class ToSet(Proc):
    def __call__(self, data: Any, **kwargs) -> set:
        return set(data)


class ToList(Proc):
    def __call__(self, data: Any, **kwargs) -> list:
        return list(data)


class ToTuple(Proc):
    def __call__(self, data: Any, **kwargs) -> tuple:
        return tuple(data)
