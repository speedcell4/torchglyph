from abc import ABCMeta
from datetime import datetime

from torch.types import Number

__all__ = [
    'Meter',
    'AverageMeter', 'ClassificationMeter', 'TimeMeter',
]


class Meter(object, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        super(Meter, self).__init__()
        self.reset(**kwargs)

    def reset(self, **kwargs) -> None:
        raise NotImplementedError


class AverageMeter(Meter):
    def reset(self, value: Number = 0, weight: Number = None) -> None:
        self.value = value
        self.weight = weight

    def update(self, value: Number, weight: Number = 1.) -> None:
        self.value += value

        if self.weight is None:
            self.weight = 0
        self.weight += weight

    @property
    def average(self) -> float:
        if self.weight is None:
            raise ValueError
        return self.value / self.weight


class ClassificationMeter(Meter):
    def reset(self, value: Number = 0, tgt_weight: Number = None, prd_weight: Number = None) -> None:
        self.value = value
        self.tgt_weight = tgt_weight
        self.prd_weight = prd_weight

    def update(self, value: Number, tgt_weight: Number, prd_weight: Number) -> None:
        self.value += value

        if self.tgt_weight is None:
            self.tgt_weight = 0
        self.tgt_weight += tgt_weight

        if self.prd_weight is None:
            self.prd_weight = 0
        self.prd_weight += prd_weight

    @property
    def precision(self) -> float:
        if self.prd_weight is None:
            raise ValueError
        return self.value / self.prd_weight

    @property
    def recall(self) -> float:
        if self.tgt_weight is None:
            raise ValueError
        return self.value / self.tgt_weight

    @property
    def f1(self) -> float:
        if self.prd_weight is None or self.tgt_weight is None:
            raise ValueError
        return self.value * 2 / (self.prd_weight + self.tgt_weight)


class TimeMeter(Meter):
    def reset(self, seconds: float = 0, count: int = None) -> None:
        self.seconds = seconds
        self.count = count

    def tik(self) -> None:
        self.start_tm = datetime.now()

    def tok(self, count: int = 1) -> None:
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        del self.start_tm

        if self.count is None:
            self.count = 0
        self.count += count

    def __enter__(self):
        self.tik()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tok(count=1)

    @property
    def average(self) -> float:
        if self.count is None:
            raise ValueError
        return self.seconds / self.count
