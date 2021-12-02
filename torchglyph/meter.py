from abc import ABCMeta
from logging import getLogger
from typing import Any, Dict

logger = getLogger(__name__)

__all__ = [
]


class Meter(object, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(Meter, self).__init__()
        self.reset_buffers()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ''

    def reset_buffers(self) -> None:
        raise NotImplementedError

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError

    @property
    def merit(self):
        raise NotImplementedError

    def __eq__(self, other: 'Meter') -> bool:
        return self.merit == other.merit

    def __le__(self, other: 'Meter') -> bool:
        return self.merit < other.merit

    def __gt__(self, other: 'Meter') -> bool:
        return self.merit > other.merit

    def state_dict(self, name: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError


class AverageMeter(Meter):
    def __init__(self, num_digits: int = 6) -> None:
        super(AverageMeter, self).__init__()
        self.num_digits = num_digits

    def extra_repr(self) -> str:
        return f'num_digits={self.num_digits}'

    def reset_buffers(self) -> None:
        self.accumulation = 0.
        self.weight = 0.

    def update(self, value: float, weight: float) -> None:
        self.accumulation += value
        self.weight += weight

    @property
    def merit(self) -> float:
        try:
            return round(self.accumulation / self.weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'{self} is empty')
            return 0.

    def state_dict(self, name: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[name] = self.merit
        return destination
