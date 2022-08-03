import logging
from collections import Counter
from typing import Tuple, Any, Set, List, Union

from torchglyph.proc.abc import Proc
from torchglyph.vocab import Vocab, PreTrainedEmbedding

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
        args = []
        if len(self.special_tokens) > 0:
            args.append(', '.join(self.special_tokens))

        args.append(f'max_size={self.max_size}')
        args.append(f'min_freq={self.min_freq}')
        return ', '.join(args)

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
    def __init__(self, n: int = None) -> None:
        super(StatsVocab, self).__init__()
        self.n = n

    def extra_repr(self) -> str:
        return f'{self.n}'

    def __call__(self, vocab: Vocab, name: str, *args, **kwargs) -> Vocab:
        logger.info(f"{name}.vocab => {vocab} :: {sum(vocab.counter.values()) / max(1, len(vocab)) :.1f} times/token")

        items = list(vocab.counter.most_common())
        if self.n is None:
            xs = ', '.join([f"'{token}' ({freq})" for token, freq in items])
            logger.info(f"{name}.tokens => [{xs}]")
        else:
            xs = ', '.join(f"'{token}' ({freq})" for token, freq in items[:+self.n // 2])
            ys = ', '.join(f"'{token}' ({freq})" for token, freq in items[-self.n // 2:])
            logger.info(f"{name}.tokens => [{xs}, ..., {ys}]")

        return vocab


class LoadVectors(Proc):
    def __init__(self, *transforms, embedding: PreTrainedEmbedding) -> None:
        super(LoadVectors, self).__init__()
        self.embedding = embedding
        self.transforms = transforms

    def extra_repr(self) -> str:
        return ', '.join([
            self.embedding.__class__.__name__,
            *[f'{fn.__name__}' for fn in self.transforms],
        ])

    def __call__(self, vocab: Vocab, *, name: str, **kwargs) -> Vocab:
        assert vocab is not None, f"did you forget '{BuildVocab.__name__}' before '{LoadVectors.__name__}'?"

        vocab.load_weight(*self.transforms, embedding=self.embedding)
        return vocab
