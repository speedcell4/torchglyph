from typing import Any, Dict, List, Sequence, Tuple

import torch
from sacremoses import MosesTokenizer
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric, MetricCollection, MinMetric, text
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
    def __init__(self, unk_token_id: int = -100) -> None:
        super(SeqMetric, self).__init__({
            'spb': MeanMetric(),  # number of sentences per batch
            'tpb': MeanMetric(),  # number of tokens per batch
            'tps': MeanMetric(),  # number of tokens per sentence
            'min': MinMetric(),
            'max': MaxMetric(),
            'unk': MeanMetric(),
        })
        self.unk_token_id = unk_token_id

    def update(self, sequence: Z) -> None:
        data, token_sizes = sequence.idx().cat()

        self['spb'].update(token_sizes.size()[0])
        self['tpb'].update(token_sizes.sum())
        self['tps'].update(token_sizes)
        self['min'].update(token_sizes)
        self['max'].update(token_sizes)
        self['unk'].update((data == self.unk_token_id).float() * 100)


class HashMetric(MetricCollection):
    def __init__(self) -> None:
        super(HashMetric, self).__init__({
            'same': MeanMetric(),
            'diff': MeanMetric(),
            'code': MeanMetric(),  # number of unique codes per batch
            'target': MeanMetric(),  # number of unique tokens per batch
        })

    def update(self, l1: Tensor, l2s: Tensor, t1: Tensor, t2s: Tensor) -> None:
        self['same'].update((l1[:, None] == l2s[None, :])[t1[:, None] == t2s[None, :]].float())
        self['diff'].update((l1[:, None] != l2s[None, :])[t1[:, None] != t2s[None, :]].float())

        n, *_ = torch.unique(l1, dim=0).size()
        m, *_ = torch.unique(t1, dim=0).size()
        self['code'].update(n)
        self['target'].update(m)

    def compute(self):
        info = super(HashMetric, self).compute()

        return {
            'same': info['same'] * 100.,
            'diff': info['diff'] * 100.,
            'code': info['code'],
            'target': info['target'],
            'unique': info['code'] * 100. / info['target'],
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


class CHRFScore(text.CHRFScore):
    def __init__(self):
        super(CHRFScore, self).__init__()

    def compute(self) -> Tensor:
        return super(CHRFScore, self).compute() * 100.


class BLEUScore(text.BLEUScore):
    def __init__(self, lang: str):
        super(BLEUScore, self).__init__()
        self.tokenizer = MosesTokenizer(lang=lang)

    def update(self, preds: Sequence[str], target: Sequence[Sequence[str]]) -> None:
        return super(BLEUScore, self).update(
            preds=[
                self.tokenizer.tokenize(t, aggressive_dash_splits=True, return_str=True)
                for t in preds
            ],
            target=[
                [self.tokenizer.tokenize(t, aggressive_dash_splits=True, return_str=True) for t in ts]
                for ts in target
            ],
        )

    def compute(self) -> Tensor:
        return super(BLEUScore, self).compute() * 100.


class SacreBLEUScore(text.SacreBLEUScore):
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
