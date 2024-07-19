import torch
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchrua import Z


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

    def update(self, sequence: Z) -> None:
        _, token_sizes = sequence.idx().cat()

        self['snt'].update(token_sizes.size()[0])
        self['tok'].update(token_sizes.sum())
        self['len'].update(token_sizes)
        self['min'].update(token_sizes)
        self['max'].update(token_sizes)


class HashMetric(MetricCollection):
    def __init__(self) -> None:
        super(HashMetric, self).__init__({
            'loss': MeanMetric(),
            'pos': MeanMetric(),
            'neg': MeanMetric(),
            'unique': MeanMetric(),
        })

    def update_loss(self, loss: Tensor) -> None:
        self['loss'].update(loss)

    def update(self, x: Tensor, y: Tensor, t1: Tensor, t2: Tensor) -> None:
        pos = t1[:, None] == t2[None, :]
        neg = t1[:, None] != t2[None, :]

        self['pos'].update((x[:, None] == y[None, :])[pos].float())
        self['neg'].update((x[:, None] != y[None, :])[neg].float())

        x, *_ = torch.unique(x, dim=0).size()
        y, *_ = torch.unique(y, dim=0).size()
        t1, *_ = torch.unique(t1, dim=0).size()
        t2, *_ = torch.unique(t2, dim=0).size()

        self['unique'].update(x / t1, t1)
        self['unique'].update(y / t2, t2)
