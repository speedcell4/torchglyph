from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric
from torchmetrics.text import SacreBLEUScore as _SacreBLEUScore
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
            'pos': MeanMetric(),
            'neg': MeanMetric(),
            'unique': MeanMetric(),
        })

    def update(self, x: Tensor, y: Tensor, t1: Tensor, t2: Tensor) -> None:
        self['pos'].update((x[:, None] == y[None, :])[t1[:, None] == t2[None, :]].float())
        self['neg'].update((x[:, None] != y[None, :])[t1[:, None] != t2[None, :]].float())

        n, *_ = torch.unique(x, dim=0).size()
        m, *_ = torch.unique(t1, dim=0).size()
        self['unique'].update(n / m, m)

        n, *_ = torch.unique(y, dim=0).size()
        m, *_ = torch.unique(t2, dim=0).size()
        self['unique'].update(n / m, m)

    def compute(self):
        return {
            key: value * 100.
            for key, value in super(HashMetric, self).compute().items()
        }


class Accuracy(MeanMetric):
    def __init__(self, ignore_index: int = -100) -> None:
        super(Accuracy, self).__init__()
        self.ignore_index = ignore_index

    def update(self, prediction: Tensor, target: Tensor) -> None:
        mask = prediction == target
        return super(Accuracy, self).update(mask.float()[target != self.ignore_index])

    def compute(self) -> Tensor:
        return super(Accuracy, self).compute() * 100.


class SacreBLEUScore(_SacreBLEUScore):
    def __init__(self, lang: str):
        tokenize = {
            'zh': 'zh',
            'ja': 'ja-mecab',
            'ko': 'ko-mecab',
        }
        super(SacreBLEUScore, self).__init__(tokenize=tokenize.get(lang, '13a'))

    def compute(self) -> Tensor:
        return super(SacreBLEUScore, self).compute() * 100.


class MultiSacreBLEUScore(MetricCollection):
    def __init__(self, lang_pairs: List[Tuple[str, str]]):
        super(MultiSacreBLEUScore, self).__init__({
            f'{lang1}-{lang2}': SacreBLEUScore(lang=lang2)
            for lang1, lang2 in lang_pairs
        })

    def compute(self) -> Dict[str, Any]:
        info = {**super(MultiSacreBLEUScore, self).compute()}
        e2x, x2e, zero, avg = [], [], [], []
        for lang_pairs, score in info.items():
            lang1, lang2 = lang_pairs.split('-')

            if lang1 == 'en':
                e2x.append(score)
            elif lang2 == 'en':
                x2e.append(score)
            else:
                zero.append(score)

            avg.append(score)

        if len(e2x) > 0:
            info['e2x'] = sum(e2x) / len(e2x)

        if len(x2e) > 0:
            info['x2e'] = sum(x2e) / len(x2e)

        if len(zero) > 0:
            info['zero'] = sum(zero) / len(zero)

        if len(avg) > 0:
            info['avg'] = sum(avg) / len(avg)

        return info
