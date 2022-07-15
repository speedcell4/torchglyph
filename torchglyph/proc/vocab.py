import heapq
import logging
from collections import Counter
from typing import Tuple, Any, Set, List, Union

from torchglyph.proc.abc import Proc
from torchglyph.vocab import Vocab, Vectors, Glove, FastText

logger = logging.getLogger(__name__)


class CountToken(Proc):
    def __call__(self, data: Any, *, counter: Counter, **kwargs) -> Any:
        counter[data] += 1
        return data


class CountTokenList(Proc):
    Data = Union[Set[Any], List[Any], Tuple[Any, ...]]

    def __call__(self, data: Data, *, counter: Counter, **kwargs) -> Data:
        counter.update(data)
        return data


class ToIndex(Proc):
    def __call__(self, data: Any, *, vocab: Vocab, **kwargs) -> int:
        return vocab[data]


class ToIndexList(Proc):
    Data = Union[Set[Any], List[Any], Tuple[Any, ...]]
    IntSequence = Union[Set[int], List[int], Tuple[int, ...]]

    def __call__(self, data: Data, *, vocab: Vocab, **kwargs) -> IntSequence:
        return type(data)([vocab[datum] for datum in data])


class BuildVocab(Proc):
    def __init__(self, unk_token: str = None, pad_token: str = None,
                 special_tokens: Tuple[str, ...] = (), max_size: int = None, min_freq: int = 0) -> None:
        super(BuildVocab, self).__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = tuple(
            token for token in (unk_token, pad_token, *special_tokens)
            if token is not None
        )

        self.max_size = max_size
        self.min_freq = min_freq

    def extra_repr(self) -> str:
        return ', '.join([
            ', '.join(self.special_tokens),
            f'max_size={self.max_size}',
            f'min_freq={self.min_freq}',
        ])

    def __call__(self, vocab: Counter, **kwargs) -> Vocab:
        return Vocab(
            counter=vocab,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )


class StatsVocab(Proc):
    def __init__(self, threshold: int) -> None:
        super(StatsVocab, self).__init__()
        self.threshold = threshold

    def extra_repr(self) -> str:
        return f'{self.threshold}'

    def __call__(self, vocab: Vocab, name: str, *args, **kwargs) -> Vocab:
        num_tokens = len(vocab.freq)
        avg_freq = sum(vocab.freq.values()) / max(1, num_tokens)

        if num_tokens <= self.threshold:
            xs = vocab.freq.most_common()
            xs = ', '.join([f"'{token}'({freq})" for token, freq in xs])

            logger.info(f"{name}.vocab => {vocab} :: {avg_freq:.1f} times/token")
            logger.info(f"{name}.tokens => [{xs}]")

        else:
            n, m = self.threshold // 2, (self.threshold + 1) // 2
            xs = heapq.nlargest(n, vocab.freq.items(), key=lambda x: x[1])
            ys = heapq.nsmallest(m, vocab.freq.items(), key=lambda x: x[1])
            xs = ', '.join([f"'{token}'({freq})" for token, freq in xs])
            ys = ', '.join([f"'{token}'({freq})" for token, freq in ys[::-1]])

            logger.info(f"{name}.vocab => {vocab} :: {avg_freq:.1f} times/token")
            logger.info(f"{name}.tokens => [{xs}, ..., {ys}]")

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
