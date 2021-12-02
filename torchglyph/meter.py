from abc import ABCMeta
from datetime import datetime
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
    def merit(self) -> float:
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
        self.total = 0.
        self.weight = 0.

    def update(self, value: float, weight: float) -> None:
        self.total += value
        self.weight += weight

    @property
    def average(self) -> float:
        try:
            return round(self.total / self.weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'weight is zero')
            return 0.

    @property
    def merit(self) -> float:
        return self.average

    def state_dict(self, name: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[name] = self.merit
        return destination


class AccuracyMeter(AverageMeter):
    def __init__(self, num_digits: int = 2) -> None:
        super(AccuracyMeter, self).__init__(num_digits=num_digits)

    @property
    def average(self) -> float:
        try:
            return round(self.total * 100 / self.weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'weight is zero')
            return 0.


class ClassificationMeter(Meter):
    def __init__(self, num_digits: int = 2) -> None:
        super(ClassificationMeter, self).__init__()
        self.num_digits = num_digits

    def extra_repr(self) -> str:
        return f'num_digits={self.num_digits}'

    def reset_buffers(self) -> None:
        self.total = 0.
        self.prediction_weight = 0.
        self.target_weight = 0.

    def update(self, value: float, prediction_weight: float, target_weight: float) -> None:
        self.total += value
        self.prediction_weight += prediction_weight
        self.target_weight += target_weight

    @property
    def precision(self) -> float:
        try:
            return round(self.total * 100 / self.prediction_weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'prediction_weight is zero')
            return 0.

    @property
    def recall(self) -> float:
        try:
            return round(self.total * 100 / self.target_weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'target_weight is zero')
            return 0.

    @property
    def f1(self) -> float:
        try:
            return round(self.total * 200 / (self.prediction_weight + self.target_weight), ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'prediction_weight + target_weight is zero')
            return 0.

    @property
    def merit(self) -> float:
        return self.f1

    def state_dict(self, name: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[f'{name}.precision'] = self.precision
        destination[f'{name}.recall'] = self.recall
        destination[f'{name}.f1'] = self.f1
        return destination


class TimeMeter(Meter):
    def __init__(self, num_digits: int = 5) -> None:
        super(TimeMeter, self).__init__()
        self.num_digits = num_digits

    def reset_buffers(self) -> None:
        self.total_seconds = 0
        self.num_units = 0

    def update(self, seconds: float, num_units: float) -> None:
        self.total_seconds += seconds
        self.num_units += num_units

    def tik(self) -> None:
        self.start_tm = datetime.now()

    def tok(self, num_units: float = 1.) -> None:
        self.total_seconds += (datetime.now() - self.start_tm).total_seconds()
        self.num_units += num_units
        del self.start_tm

    def __enter__(self):
        self.tik()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tok()

    @property
    def units_per_second(self) -> float:
        try:
            return round(self.num_units / self.total_seconds, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.info(f'{self.total_seconds} is zero')
            return 0.

    @property
    def seconds_per_unit(self) -> float:
        try:
            return round(self.total_seconds / self.num_units, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.info(f'{self.num_units} is zero')
            return 0.

    @property
    def merit(self) -> float:
        return self.units_per_second

    def state_dict(self, name: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[f'{name}.units_per_second'] = self.units_per_second
        destination[f'{name}.seconds_per_unit'] = self.seconds_per_unit
        return destination
