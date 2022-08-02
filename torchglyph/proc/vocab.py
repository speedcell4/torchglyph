import heapq
import logging
from collections import Counter
from typing import Tuple, Any, Set, List, Union

from torchglyph.proc.abc import Proc
from torchglyph.vocab import Vocab, PreTrainedEmbedding, Glove, FastText

logger = logging.getLogger(__name__)


class CountToken(Proc):
    def __call__(self, data: Any, *, counter: Counter, **kwargs) -> Any:
        counter[data] += 1
        return data


class CountTokenSequence(Proc):
    Data = Union[Set[Any], List[Any], Tuple[Any, ...]]

    def __call__(self, data: Data, *, counter: Counter, **kwargs) -> Data:
        counter.update(data)
        return data


class ToIndex(Proc):
    def __call__(self, data: Any, *, vocab: Vocab, **kwargs) -> int:
        return vocab[data]


class ToIndexSequence(Proc):
    Data = Union[Set[Any], List[Any], Tuple[Any, ...]]
    IntSequence = Union[Set[int], List[int], Tuple[int, ...]]

    def __call__(self, data: Data, *, vocab: Vocab, **kwargs) -> IntSequence:
        return type(data)([vocab[datum] for datum in data])


class BuildVocab(Proc):
    def __init__(self, unk_token: str = None, pad_token: str = None,
                 bos_token: str = None, eos_token: str = None,
                 special_tokens: Tuple[str, ...] = (),
                 max_size: int = None, min_freq: int = 0) -> None:
        super(BuildVocab, self).__init__()

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.special_tokens = special_tokens

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
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            special_tokens=self.special_tokens,
        ).build(
            counter=vocab,
            max_size=self.max_size,
            min_freq=self.min_freq,
        )


class StatsVocab(Proc):
    def __init__(self, n: int) -> None:
        super(StatsVocab, self).__init__()
        self.n = n

    def extra_repr(self) -> str:
        return f'{self.n}'

    def __call__(self, vocab: Vocab, name: str, *args, **kwargs) -> Vocab:
        num_tokens = len(vocab)
        average_freq = sum(vocab.counter.values()) / max(1, num_tokens)

        if num_tokens <= self.n:
            xs = vocab.counter.most_common()
            xs = ', '.join([f"'{token}'({freq})" for token, freq in xs])

            logger.info(f"{name}.vocab => {vocab} :: {average_freq:.1f} times/token")
            logger.info(f"{name}.tokens => [{xs}]")

        else:
            n, m = self.n // 2, (self.n + 1) // 2
            xs = heapq.nlargest(n, vocab.counter.items(), key=lambda x: x[1])
            ys = heapq.nsmallest(m, vocab.counter.items(), key=lambda x: x[1])
            xs = ', '.join([f"'{token}'({freq})" for token, freq in xs])
            ys = ', '.join([f"'{token}'({freq})" for token, freq in ys[::-1]])

            logger.info(f"{name}.vocab => {vocab} :: {average_freq:.1f} times/token")
            logger.info(f"{name}.tokens => [{xs}, ..., {ys}]")

        return vocab


class LoadVectors(Proc):
    def __init__(self, *transforms, embedding: PreTrainedEmbedding) -> None:
        super(LoadVectors, self).__init__()
        self.transforms = transforms
        self.embedding = embedding

    def extra_repr(self) -> str:
        return ', '.join([
            self.embedding.__class__.__name__,
            *[f'{fn.__name__}' for fn in self.transforms],
        ])

    def __call__(self, vocab: Vocab, *, name: str, **kwargs) -> Vocab:
        assert vocab is not None, f"did you forget '{BuildVocab.__name__}' before '{LoadVectors.__name__}'?"

        vocab.load_weight(*self.transforms, embedding=self.embedding)
        return vocab


class LoadGlove(LoadVectors):
    def __init__(self, *transforms, name: str, dim: int) -> None:
        super(LoadGlove, self).__init__(
            *transforms, embedding=Glove(name=name, dim=dim),
        )


class LoadFastText(LoadVectors):
    def __init__(self, *transforms, name: str, lang: str) -> None:
        super(LoadFastText, self).__init__(
            *transforms, embedding=FastText(name=name, lang=lang),
        )
