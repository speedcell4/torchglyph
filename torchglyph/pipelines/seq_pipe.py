from typing import Union

from torchglyph.dataset import Pipeline
from torchglyph.proc import AddToCounter, BuildVocab, ToTensor, ToRange, LoadGlove
from torchglyph.proc import Numbering, PackSeq, PadSeq
from torchglyph.proc import Scan


class PaddedSeqPipe(Pipeline):
    def __init__(self, pad_token: Union[str, int], batch_first: bool = True, dim: int = None) -> None:
        super(PaddedSeqPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(special_tokens=(pad_token,)) + (LoadGlove('6B', dim) if dim is not None else None),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PadSeq(pad_token, batch_first),
        )


class PackedSeqPipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSeqPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PackSeq(),
        )


class PackedSeqRangePipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSeqRangePipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToRange() + ToTensor(),
            batch_procs=Scan(lambda t, a: (t + a, t.size(0) + a), 0) + PackSeq(),
        )
