from collections import Counter
from typing import Any, Tuple

from torchglyph.proc import Flatten, Proc
from torchglyph.proc.utiles import stoi
from torchglyph.vocab import Vocab, Vectors, Glove


class AddToCounter(Proc):
    @classmethod
    def obtain_tokens(cls, ins):
        if isinstance(ins, str):
            yield ins
        else:
            for x in ins:
                yield from cls.obtain_tokens(x)

    def __call__(self, ins, counter: Counter, **kwargs) -> Any:
        counter.update(self.obtain_tokens(ins))
        return ins


class BuildVocab(Proc):
    def __init__(self, unk_token: str = '<unk>', pad_token: str = None,
                 special_tokens: Tuple[str, ...] = (),
                 max_size: int = None, min_freq: int = 1) -> None:
        super(BuildVocab, self).__init__()
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.special_tokens = special_tokens
        self.max_size = max_size
        self.min_freq = min_freq

    def __call__(self, vocab: Counter, **kwargs) -> Vocab:
        return Vocab(
            counter=vocab,
            unk_token=self.unk_token,
            pad_token=self.pad_token,
            special_tokens=self.special_tokens,
            max_size=self.max_size, min_freq=self.min_freq,
        )


class Numbering(Flatten):
    def process(self, token: str, vocab: Vocab, **kwargs) -> int:
        return stoi(token=token, vocab=vocab)


class LoadVectors(Proc):
    def __init__(self, vectors: Vectors) -> None:
        super(LoadVectors, self).__init__()
        self.vectors = vectors

    def __call__(self, vocab: Vocab, **kwargs) -> Vocab:
        assert vocab is not None, f"did you forget '{BuildVocab.__name__}' before '{LoadVectors.__name__}'?"

        vocab.load_vectors(self.vectors)
        return vocab


class LoadGlove(LoadVectors):
    def __init__(self, name: str, dim: int) -> None:
        super(LoadGlove, self).__init__(
            vectors=Glove(name=name, dim=dim),
        )
