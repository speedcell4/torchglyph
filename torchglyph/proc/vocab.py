import logging
from collections import Counter
from typing import Tuple, Optional

from torchglyph.proc.abc import Proc, Map
from torchglyph.vocab import Vocab, Vectors, Glove, FastText

logger = logging.getLogger(__name__)

__all__ = [
    'THRESHOLD',
    'UpdateCounter', 'Numbering', 'BuildVocab', 'StatsVocab',
    'LoadVectors', 'LoadGlove', 'LoadFastText',
]

THRESHOLD = 10


class UpdateCounter(Map):
    def map(self, token: str, *, counter: Counter, **kwargs) -> str:
        counter[token] += 1
        return token


class Numbering(Map):
    def map(self, token: str, *, vocab: Vocab, **kwargs) -> int:
        return vocab[token]


class BuildVocab(Proc):
    def __init__(self, unk_token: Optional[str], pad_token: Optional[str],
                 special_tokens: Tuple[Optional[str], ...] = ()) -> None:
        super(BuildVocab, self).__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        special_tokens = (unk_token, pad_token, *special_tokens)
        self.special_tokens = tuple(token for token in special_tokens if token is not None)

    def extra_repr(self) -> str:
        return ', '.join(self.special_tokens)

    def __call__(self, vocab: Counter, *, max_size: Optional[int], min_freq: int, **kwargs) -> Vocab:
        return Vocab(
            counter=vocab,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=max_size, min_freq=min_freq,
        )


class StatsVocab(Proc):
    def __init__(self, threshold: int = THRESHOLD) -> None:
        super(StatsVocab, self).__init__()
        self.threshold = threshold

    def extra_repr(self) -> str:
        return f'{self.threshold}'

    def __call__(self, vocab: Vocab, name: str, *args, **kwargs) -> Vocab:
        assert vocab is not None
        assert name is not None

        min_tok, min_occ = min(vocab.freq.items(), key=lambda x: x[1])
        max_tok, max_occ = max(vocab.freq.items(), key=lambda x: x[1])
        cnt_tok = len(vocab.freq.values())
        avg_occ = sum(vocab.freq.values()) / max(1, cnt_tok)

        name = f"{vocab.__class__.__name__} '{name}'"
        logger.info(f"{name} has {cnt_tok} token(s) => "
                    f"{avg_occ:.1f} times/token ["
                    f"{max_occ} :: '{max_tok}', "
                    f"{min_occ} :: '{min_tok}']")
        if cnt_tok <= self.threshold:
            logger.info(f'{name} => [{", ".join(vocab.itos)}]')
        else:
            logger.info(f'{name} => ['
                        f'{", ".join(vocab.itos[:self.threshold // 2])}, ..., '
                        f'{", ".join(vocab.itos[-self.threshold // 2:])}]')

        return vocab


class LoadVectors(Proc):
    def __init__(self, *fallbacks, vectors: Vectors) -> None:
        super(LoadVectors, self).__init__()
        self.fallbacks = fallbacks
        self.vectors = vectors

    def extra_repr(self) -> str:
        return ', '.join([
            self.vectors.__class__.__name__,
            *[f'{fallback.__name__}' for fallback in self.fallbacks],
        ])

    def __call__(self, vocab: Vocab, *, name: str, **kwargs) -> Vocab:
        assert vocab is not None, \
            f"did you forget '{BuildVocab.__name__}' before '{LoadVectors.__name__}'?"

        tok, occ = vocab.load_vectors(*self.fallbacks, vectors=self.vectors)
        tok = tok / max(1, len(vocab.freq.values())) * 100
        occ = occ / max(1, sum(vocab.freq.values())) * 100

        logger.info(f"{self.vectors} hits {tok:.1f}% tokens "
                    f"and {occ:.1f}% occurrences of {Vocab.__name__} '{name}'")
        return vocab


class LoadGlove(LoadVectors):
    def __init__(self, *fallbacks, name: str, dim: int) -> None:
        super(LoadGlove, self).__init__(
            *fallbacks, vectors=Glove(name=name, dim=dim),
        )


class LoadFastText(LoadVectors):
    def __init__(self, *fallbacks, name: str, lang: str) -> None:
        super(LoadFastText, self).__init__(
            *fallbacks, vectors=FastText(name=name, lang=lang),
        )
