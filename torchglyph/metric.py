from typing import Union

from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchrua import C, D, P


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
