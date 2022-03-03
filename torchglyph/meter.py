from abc import ABCMeta
from datetime import datetime
from logging import getLogger
from typing import Any, Dict

logger = getLogger(__name__)

__all__ = [
    'Meter',
    'AverageMeter', 'AccuracyMeter',
    'ClassificationMeter',
    'TimeMeter',
]


def join(*names: str, sep: str = '.') -> str:
    return sep.join(name for name in names if len(name) > 0)


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

    def __lt__(self, other: 'Meter') -> bool:
        return self.merit < other.merit

    def __gt__(self, other: 'Meter') -> bool:
        return self.merit > other.merit

    def __ne__(self, other: 'Meter') -> bool:
        return self.merit != other.merit

    def __le__(self, other: 'Meter') -> bool:
        return self.merit <= other.merit

    def __ge__(self, other: 'Meter') -> bool:
        return self.merit >= other.merit

    def __add__(self, other: 'Meter') -> 'Meter':
        raise NotImplementedError

    def state_dict(self, prefix: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def flush(self, prefix: str, reset_buffers: bool = True) -> None:
        for key, value in self.state_dict(prefix=prefix, destination=None).items():
            logger.info(f'{key} => {value}')

        if reset_buffers:
            self.reset_buffers()


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

    def __add__(self, other: 'AverageMeter') -> 'AverageMeter':
        meter = self.__class__(num_digits=self.num_digits)
        meter.update(value=self.total, weight=self.weight)
        meter.update(value=other.total, weight=other.weight)
        return meter

    @property
    def average(self) -> float:
        try:
            return round(self.total / self.weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'{self.weight} is zero')
            return 0.

    @property
    def merit(self) -> float:
        return self.average

    def state_dict(self, prefix: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[prefix] = self.merit
        return destination


class AccuracyMeter(AverageMeter):
    def __init__(self, num_digits: int = 2) -> None:
        super(AccuracyMeter, self).__init__(num_digits=num_digits)

    @property
    def average(self) -> float:
        try:
            return round(self.total * 100 / self.weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'{self.weight} is zero')
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

    def __add__(self, other: 'ClassificationMeter') -> 'ClassificationMeter':
        meter = self.__class__(num_digits=self.num_digits)
        meter.update(value=self.total, prediction_weight=self.prediction_weight, target_weight=self.target_weight)
        meter.update(value=other.total, prediction_weight=other.prediction_weight, target_weight=other.target_weight)
        return meter

    @property
    def precision(self) -> float:
        try:
            return round(self.total * 100 / self.prediction_weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'{self.prediction_weight} is zero')
            return 0.

    @property
    def recall(self) -> float:
        try:
            return round(self.total * 100 / self.target_weight, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'{self.target_weight} is zero')
            return 0.

    @property
    def f1(self) -> float:
        try:
            return round(self.total * 200 / (self.prediction_weight + self.target_weight), ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.warning(f'{self.prediction_weight + self.target_weight} is zero')
            return 0.

    @property
    def merit(self) -> float:
        return self.f1

    def state_dict(self, prefix: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[join(prefix, 'precision')] = self.precision
        destination[join(prefix, 'recall')] = self.recall
        destination[join(prefix, 'f1')] = self.f1
        return destination


class TimeMeter(Meter):
    def __init__(self, num_digits: int = 6) -> None:
        super(TimeMeter, self).__init__()
        self.num_digits = num_digits

    def reset_buffers(self) -> None:
        self.seconds = 0
        self.num_units = 0

    def update(self, seconds: float, num_units: float) -> None:
        self.seconds += seconds
        self.num_units += num_units

    def __add__(self, other: 'TimeMeter') -> 'TimeMeter':
        meter = self.__class__(num_digits=self.num_digits)
        meter.update(seconds=self.seconds, num_units=self.num_units)
        meter.update(seconds=other.seconds, num_units=other.num_units)
        return meter

    def tik(self) -> None:
        self.start_datetime = datetime.now()

    def tok(self, num_units: float = 1.) -> None:
        self.seconds += (datetime.now() - self.start_datetime).total_seconds()
        self.num_units += num_units
        del self.start_datetime

    def __enter__(self):
        self.tik()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tok()

    @property
    def units_per_second(self) -> float:
        try:
            return round(self.num_units / self.seconds, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.info(f'{self.seconds} is zero')
            return 0.

    @property
    def seconds_per_unit(self) -> float:
        try:
            return round(self.seconds / self.num_units, ndigits=self.num_digits)
        except ZeroDivisionError:
            logger.info(f'{self.num_units} is zero')
            return 0.

    @property
    def merit(self) -> float:
        return self.units_per_second

    def state_dict(self, prefix: str, *, destination: Dict[str, Any] = None) -> Dict[str, Any]:
        if destination is None:
            destination = {}

        destination[join(prefix, 'units_per_second')] = self.units_per_second
        destination[join(prefix, 'seconds_per_unit')] = self.seconds_per_unit
        return destination
