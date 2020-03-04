from torchglyph.dataset import Pipeline
from torchglyph.processes import AddToCounter, BuildVocab, ToTensor, ToChar, ToRange, PackAccSeqBatch, ToLength, \
    PadTokBatch, Numbering, PackArrayBatch, ToTensorList, PackSeqBatch


class SeqPackPipe(Pipeline):
    def __init__(self) -> None:
        super(SeqPackPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PackSeqBatch(),
        )


class SeqLengthPipe(Pipeline):
    def __init__(self) -> None:
        super(SeqLengthPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToLength(),
            batch_procs=PadTokBatch(),
        )


class SeqRangePipe(Pipeline):
    def __init__(self) -> None:
        super(SeqRangePipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToRange() + ToTensor(),
            batch_procs=PackAccSeqBatch(),
        )


class CharArrayPackPipe(Pipeline):
    def __init__(self) -> None:
        super(CharArrayPackPipe, self).__init__(
            pre_procs=ToChar() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensorList(),
            batch_procs=PackArrayBatch(),
        )
