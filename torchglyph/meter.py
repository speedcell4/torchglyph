from logging import getLogger
from numbers import Number
from pathlib import Path
from typing import Tuple, Union

import torch
from torch import Tensor, nn
from torchmetrics import MaxMetric, MeanMetric, Metric, MetricCollection, MinMetric
from torchrua import C, D, P

from torchglyph.dist import is_master
from torchglyph.serde import save_sota

logger = getLogger(__name__)


def detach(obj, *, n: int = 2):
    if isinstance(obj, float):
        return round(obj, ndigits=n)

    if torch.is_tensor(obj):
        return detach(obj.detach().cpu().item(), n=n)

    if isinstance(obj, (str, bool, int, complex)):
        return obj

    if isinstance(obj, (set, list, tuple)):
        return type(obj)(detach(o, n=n) for o in obj)

    if isinstance(obj, dict):
        return {k: detach(v, n=n) for k, v in obj.items()}

    raise TypeError(f'{type(obj)} is not supported')


class Meter(nn.Module):
    def stats(self) -> Tuple[Union[Number, Tensor], ...]:
        raise NotImplementedError

    def __gt__(self, other: 'Meter') -> bool:
        return detach(self.stats()) > detach(other.stats())

    def __lt__(self, other: 'Meter') -> bool:
        return detach(self.stats()) < detach(other.stats())

    def __eq__(self, other: 'Meter') -> bool:
        return detach(self.stats()) == detach(other.stats())

    def __ge__(self, other: 'Meter') -> bool:
        return detach(self.stats()) >= detach(other.stats())

    def __le__(self, other: 'Meter') -> bool:
        return detach(self.stats()) <= detach(other.stats())

    def __ne__(self, other: 'Meter') -> bool:
        return detach(self.stats()) != detach(other.stats())

    def reset(self) -> None:
        for metric in self._modules.values():
            if isinstance(metric, (Metric, MetricCollection)):
                metric.reset()

    def log(self, desc: str, step: int, out_dir: Path = None) -> None:
        for name, metric in self._modules.items():
            if isinstance(metric, (Metric, MetricCollection)):
                values = detach(metric.compute())

                if not isinstance(values, dict):
                    values = {name: values}
                else:
                    values = {f'{name}-{k}': v for k, v in values.items()}

                msg = ' | '.join(f'{key} {value}' for key, value in values.items())
                logger.info(f'{desc} {step} => {msg}')

                if out_dir is not None and is_master():
                    save_sota(out_dir=out_dir, step=step, **{
                        f'{desc}-{key}': value
                        for key, value in values.items()
                    })


class TensorMetric(MetricCollection):
    def __init__(self) -> None:
        super(TensorMetric, self).__init__({
            'abs': MeanMetric(),
            'avg': MeanMetric(),
            'min': MinMetric(),
            'max': MaxMetric(),
        })

    def update(self, tensor: Tensor) -> None:
        self['abs'].update(tensor.abs())
        self['avg'].update(tensor)
        self['min'].update(tensor)
        self['max'].update(tensor)


class SeqMetric(MetricCollection):
    def __init__(self) -> None:
        super(SeqMetric, self).__init__({
            'snt': MeanMetric(),
            'tok': MeanMetric(),
            'len': MeanMetric(),
            'min': MinMetric(),
            'max': MaxMetric(),
        })

    def update(self, sequence: Union[C, D, P]) -> None:
        _, token_sizes = sequence.idx().cat()

        self['snt'].update(token_sizes.size()[0])
        self['tok'].update(token_sizes.sum())
        self['len'].update(token_sizes)
        self['min'].update(token_sizes)
        self['max'].update(token_sizes)


class AccuracyMetric(MeanMetric):
    def update(self, argmax: Tensor, target: Tensor, ignore_index: int = -100):
        if ignore_index is not None:
            mask = target != ignore_index
            argmax = argmax[mask]
            target = target[mask]

        return super(AccuracyMetric, self).update((argmax == target) * 100.)
