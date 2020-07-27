import logging
from collections import Counter
from typing import Tuple, List, Union, Optional

from torchglyph.proc import Proc
from torchglyph.vocab import Vocab, Vectors, Glove, FastTest

logger = logging.getLogger(__name__)


class UpdateCounter(Proc):
    def __call__(self, data: Union[str, List[str]], counter: Counter, *args, **kwargs) -> Union[str, List[str]]:
        if isinstance(data, str):
            counter[data] += 1
        else:
            counter.update(data)
        return data


class BuildVocab(Proc):
    def __init__(self, unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = ()) -> None:
        super(BuildVocab, self).__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens

    def extra_repr(self) -> str:
        args = ', '.join(set([
            f"'{t}'" for t in (self.unk_token, self.pad_token, *self.special_tokens)
            if t is not None
        ]))
        if len(args) == 0:
            return ''
        return f'with={args}'

    def __call__(self, vocab: Counter, *args, special_tokens: Tuple[str, ...],
                 max_size: Optional[int], min_freq: int, **kwargs) -> Vocab:
        return Vocab(
            counter=vocab,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=(*self.special_tokens, *special_tokens),
            max_size=max_size, min_freq=min_freq,
        )


class StatsVocab(Proc):
    def __init__(self, threshold: int) -> None:
        super(StatsVocab, self).__init__()
        self.threshold = threshold

    def extra_repr(self) -> str:
        return f'{self.threshold}'

    def __call__(self, vocab: Vocab, name: str, *args, **kwargs) -> Vocab:
        assert vocab is not None
        assert name is not None

        tok_min, occ_min = min(vocab.freq.items(), key=lambda x: x[1])
        tok_max, occ_max = max(vocab.freq.items(), key=lambda x: x[1])
        tok_cnt = len(vocab.freq.values())
        occ_avg = sum(vocab.freq.values()) / max(1, tok_cnt)

        name = f"{vocab.__class__.__name__} '{name}'"
        logger.info(f"{name} has {tok_cnt} token(s) => "
                    f"{occ_avg:.1f} occurrence(s)/token ["
                    f"{occ_max} :: '{tok_max}', "
                    f"{occ_min} :: '{tok_min}']")
        if tok_cnt <= self.threshold:
            logger.info(f'{name} => [{", ".join(vocab.itos)}]')
        else:
            logger.info(f'{name} => ['
                        f'{", ".join(vocab.itos[:self.threshold // 2])}, ..., '
                        f'{", ".join(vocab.itos[-self.threshold // 2:])}]')

        return vocab


class LoadVectors(Proc):
    def __init__(self, *fallback_fns, vectors: Vectors, remove_missing: bool) -> None:
        super(LoadVectors, self).__init__()
        self.fallback_fns = fallback_fns
        self.vectors = vectors
        self.remove_missing = remove_missing

    def extra_repr(self) -> str:
        return ', '.join([
            *[f'{f.__name__}' for f in self.fallback_fns],
            f'{self.vectors.extra_repr()}',
            f'remove_missing={self.remove_missing}',
        ])

    def __call__(self, vocab: Vocab, name: str, *args, **kwargs) -> Vocab:
        assert vocab is not None, f"did you forget '{BuildVocab.__name__}' before '{LoadVectors.__name__}'?"

        if self.remove_missing:
            vocab = vocab.union(self.vectors, *self.fallback_fns)
        tok, occ = vocab.load_vectors(*self.fallback_fns, vectors=self.vectors)
        tok = tok / max(1, len(vocab.freq.values())) * 100
        occ = occ / max(1, sum(vocab.freq.values())) * 100
        logger.info(f"{self.vectors} hits {tok:.1f}% tokens and {occ:.1f}% occurrences of {Vocab.__name__} '{name}'")
        return vocab


class LoadGlove(LoadVectors):
    def __init__(self, *fallback_fns, name: str, dim: int, remove_missing: bool) -> None:
        super(LoadGlove, self).__init__(
            *fallback_fns,
            vectors=Glove(name=name, dim=dim),
            remove_missing=remove_missing,
        )


class LoadFastText(LoadVectors):
    def __init__(self, *fallback_fns, lang: str, remove_missing: bool) -> None:
        super(LoadFastText, self).__init__(
            *fallback_fns,
            vectors=FastTest(lang=lang),
            remove_missing=remove_missing,
        )
