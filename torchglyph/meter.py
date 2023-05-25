import datetime
from dataclasses import dataclass, field
from datetime import datetime
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, get_type_hints

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torch.types import Number
from torchrua import CattedSequence

from torchglyph.dist import all_gather_object, is_master
from torchglyph.io import save_sota

logger = getLogger(__name__)

TCP = Union[Tuple, CattedSequence, PackedSequence]


def detach_tensor(method):
    def wrap(self, *args: Union[Number, Tensor]):
        detached_args = []
        for arg in args:
            if torch.is_tensor(arg):
                arg = arg.detach().cpu().item()
            detached_args.append(arg)

        return method(self, *detached_args)

    return wrap


def zero_division(default):
    def wrap1(method):
        def wrap2(self, *args):
            try:
                return method(self, *args)
            except ZeroDivisionError:
                return default

        return wrap2

    return wrap1


class Meter(object):
    @property
    def keys(self) -> Tuple[Number, ...]:
        raise NotImplementedError

    def __eq__(self, other: 'Meter') -> bool:
        return self.keys == other.keys

    def __lt__(self, other: 'Meter') -> bool:
        return self.keys < other.keys

    def __gt__(self, other: 'Meter') -> bool:
        return self.keys > other.keys

    def __ne__(self, other: 'Meter') -> bool:
        return self.keys != other.keys

    def __le__(self, other: 'Meter') -> bool:
        return self.keys <= other.keys

    def __ge__(self, other: 'Meter') -> bool:
        return self.keys >= other.keys

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        out = {}

        for name in get_type_hints(self):
            if isinstance(getattr(self, name), Meter):
                for keys, value in getattr(self, name).stats.items():
                    out[(name, *keys)] = value

            else:
                logger.critical(f'field {name} is ignored')

        return out

    def log(self, stage: str, iteration: int, out_dir: Path = None):
        for name in get_type_hints(self):
            if isinstance(getattr(self, name), Meter):
                msg = ' | '.join(
                    f"{'-'.join([name, *keys])} {value}"
                    for keys, value in getattr(self, name).stats.items()
                )
                logger.info(f'{stage} {iteration} => {msg}')

                if is_master():
                    save_sota(out_dir=out_dir, step=iteration, **{
                        '-'.join([stage, name, *keys]): value
                        for keys, value in getattr(self, name).stats.items()
                    })
            else:
                logger.critical(f'field {name} is ignored')

        return self

    def gather(self):
        for name in get_type_hints(self):
            if isinstance(getattr(self, name), Meter):
                getattr(self, name).gather()
            else:
                logger.critical(f'field {name} is ignored')

        return self

    def update(self, *args) -> None:
        raise NotImplementedError


@dataclass()
class MaxMeter(Meter):
    value: Number = -float('inf')

    @property
    def max(self) -> Number:
        return self.value

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.max,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.max}

    @detach_tensor
    def update(self, value) -> None:
        self.value = max(self.value, value)

    def gather(self) -> None:
        self.value = max(all_gather_object(self.value))


@dataclass()
class MinMeter(Meter):
    value: Number = +float('inf')

    @property
    def min(self) -> Number:
        return self.value

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.min,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.min}

    @detach_tensor
    def update(self, value) -> None:
        self.value = min(self.value, value)

    def gather(self) -> None:
        self.value = min(all_gather_object(self.value))


@dataclass()
class SumMeter(Meter):
    value: Number = 0

    @property
    def sum(self) -> Number:
        return self.value

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.sum,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.sum}

    @detach_tensor
    def update(self, value) -> None:
        self.value += value

    def gather(self) -> None:
        self.value = sum(all_gather_object(self.value))


@dataclass()
class AverageMeter(Meter):
    value: Number = 0
    weight: Number = 0

    @property
    @zero_division(default=0)
    def average(self) -> Number:
        return round(self.value / self.weight, ndigits=2)

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.average,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.average}

    @detach_tensor
    def update_by_sum(self, value, weight=1) -> None:
        self.value += value
        self.weight += weight

    @detach_tensor
    def update_by_mean(self, value, weight=1) -> None:
        self.value += value * weight
        self.weight += weight

    def gather(self) -> None:
        self.value = sum(all_gather_object(self.value))
        self.weight = sum(all_gather_object(self.weight))


@dataclass()
class AccuracyMeter(Meter):
    value: Number = 0
    weight: Number = 0

    @property
    @zero_division(default=0)
    def accuracy(self) -> Number:
        return round(self.value * 100 / self.weight, ndigits=2)

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.accuracy,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {(): self.accuracy}

    @detach_tensor
    def update_by_sum(self, value, weight=1) -> None:
        self.value += value
        self.weight += weight

    @detach_tensor
    def update_by_mean(self, value, weight=1) -> None:
        self.value += value * weight
        self.weight += weight

    def update_by_tensor(self, prd: Tensor, tgt: Tensor, pad_token_id: int):
        mask = tgt != pad_token_id
        return self.update_by_sum(((prd == tgt) & mask).long().sum(), mask.long().sum())

    def update_by_sequence(self, prd: TCP, tgt: TCP) -> None:
        assert type(prd) is type(tgt), f'{type(prd)} is not  {type(tgt)}'

        self.update_by_sum((prd.data == tgt.data).float().sum(), prd.data.numel())

    def gather(self) -> None:
        self.value = sum(all_gather_object(self.value))
        self.weight = sum(all_gather_object(self.weight))


@dataclass()
class ClassificationMeter(Meter):
    value: Number = 0
    weight_prd: Number = 0
    weight_tgt: Number = 0

    @property
    @zero_division(default=0)
    def precision(self) -> Number:
        return round(self.value * 100 / self.weight_prd, ndigits=2)

    @property
    @zero_division(default=0)
    def recall(self) -> Number:
        return round(self.value * 100 / self.weight_tgt, ndigits=2)

    @property
    @zero_division(default=0)
    def f1(self) -> Number:
        return round(self.value * 200 / (self.weight_prd + self.weight_tgt), ndigits=2)

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.f1, self.precision, self.recall

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {('f1',): self.f1, ('precision',): self.precision, ('recall',): self.recall}

    @detach_tensor
    def update(self, value, weight_prd, weight_tgt) -> None:
        self.value += value
        self.weight_prd += weight_prd
        self.weight_tgt += weight_tgt

    def update_by_prd(self, prd: List[Any], tgt: List[Any]) -> None:
        prd = set(prd)
        tgt = set(tgt)

        self.value += len(prd & tgt)
        self.weight_prd += len(prd)
        self.weight_tgt += len(tgt)

    def gather(self) -> None:
        self.value = sum(all_gather_object(self.value))
        self.weight_prd = sum(all_gather_object(self.weight_prd))
        self.weight_tgt = sum(all_gather_object(self.weight_tgt))


@dataclass()
class TimeMeter(Meter):
    seconds: float = 0
    units: int = 0

    @property
    @zero_division(default=0)
    def second_per_unit(self) -> float:
        return round(self.seconds / self.units, ndigits=6)

    @property
    @zero_division(default=0)
    def unit_per_second(self) -> float:
        return round(self.units / self.seconds, ndigits=6)

    @property
    def keys(self) -> Tuple[Number, ...]:
        return self.unit_per_second,

    @property
    def stats(self) -> Dict[Tuple[str, ...], Number]:
        return {
            ('unit_per_second',): self.unit_per_second,
            ('second_per_unit',): self.second_per_unit,
        }

    def tik(self) -> None:
        self.start_tm = datetime.now()

    def tok(self) -> None:
        self.seconds += (datetime.now() - self.start_tm).total_seconds()
        self.units += 1

        del self.start_tm

    def __enter__(self):
        self.tik()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tok()

    @detach_tensor
    def update(self, seconds: float, units: int = 1) -> None:
        self.seconds += seconds
        self.units += units

    def gather(self) -> None:
        self.seconds = sum(all_gather_object(self.seconds))
        self.units = sum(all_gather_object(self.units))


@dataclass()
class SequenceMeter(Meter):
    snt: AverageMeter = field(default_factory=AverageMeter)
    tok: AverageMeter = field(default_factory=AverageMeter)
    len: AverageMeter = field(default_factory=AverageMeter)
    max: MaxMeter = field(default_factory=MaxMeter)

    @property
    def keys(self) -> Tuple[Number, ...]:
        return 0,

    def update(self, sequence: Union[CattedSequence, PackedSequence]):
        t, s, *_ = sequence.size()
        m, *_ = sequence.data.size()

        self.snt.update_by_sum(s)
        self.tok.update_by_sum(t)
        self.len.update_by_sum(t, s)
        self.max.update(m)
