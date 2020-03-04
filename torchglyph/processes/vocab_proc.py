from collections import Counter

from torchglyph.proc import VocabProc
from torchglyph.vocab import Vocab


class BuildVocab(VocabProc):
    def __init__(self, unk_token: str = '<unk>',
                 max_size: int = None, min_freq: int = 1) -> None:
        super(BuildVocab, self).__init__()
        self.unk_token = unk_token
        self.max_size = max_size
        self.min_freq = min_freq

    def __call__(self, vocab: Counter) -> Vocab:
        return Vocab(
            counter=vocab, unk_token=self.unk_token,
            max_size=self.max_size, min_freq=self.min_freq,
        )
