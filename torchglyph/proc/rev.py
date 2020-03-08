from typing import List

from torchglyph.proc.abc import Proc
from torchglyph.vocab import Vocab


class RevVocab(Proc):
    def __call__(self, data: List[int], vocab: Vocab, *args, **kwargs) -> List[str]:
        return [vocab.itos[datum] for datum in data]
