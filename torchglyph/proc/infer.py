from typing import List

from torchglyph.proc import Proc
from torchglyph.vocab import Vocab


class RevVocab(Proc):
    def __call__(self, xs: List[int], vocab: Vocab, **kwargs) -> List[str]:
        return [vocab.itos[x] for x in xs]
