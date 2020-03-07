from collections import Counter
from typing import Tuple, List, Union

from torchglyph.proc.abc import Recur, Proc
from torchglyph.proc.utilities import stoi
from torchglyph.vocab import Vocab, Vectors, Glove


class UpdateCounter(Proc):
    def __call__(self, data: Union[str, List[str]], counter: Counter, **kwargs) -> Union[str, List[str]]:
        if isinstance(data, str):
            data = (data,)
        counter.update(data)
        return data


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


class Numbering(Recur):
    def process(self, datum: str, vocab: Vocab, **kwargs) -> int:
        return stoi(token=datum, vocab=vocab)


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
