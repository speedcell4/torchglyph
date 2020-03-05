from typing import Union

from torchglyph.dataset import Pipeline
from torchglyph.processes import AddToCounter, BuildVocab, ToTensor, ToRange, LoadGlove
from torchglyph.processes import PackAccSeqBatch, Numbering, PackSeqBatch, PadSeqBatch


class PaddedSeqPipe(Pipeline):
    def __init__(self, pad_token: Union[str, int], batch_first: bool = True, dim: int = None) -> None:
        super(PaddedSeqPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(special_tokens=(pad_token,)) + (LoadGlove('6B', dim) if dim is not None else None),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PadSeqBatch(pad_token, batch_first),
        )


class PackedSeqPipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSeqPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PackSeqBatch(),
        )


class PackedSeqRangePipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSeqRangePipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToRange() + ToTensor(),
            batch_procs=PackAccSeqBatch(),
        )
