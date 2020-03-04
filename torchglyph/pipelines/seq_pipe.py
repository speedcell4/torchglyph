from torchglyph.dataset import Pipeline
from torchglyph.processes import AddToCounter, BuildVocab, Numbering, ToTensor, PackSeqBatch, ToChar, ToTensorList, \
    PackArrayBatch


class SeqPackPipe(Pipeline):
    def __init__(self) -> None:
        super(SeqPackPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PackSeqBatch(),
        )


class CharArrayPackPipe(Pipeline):
    def __init__(self) -> None:
        super(CharArrayPackPipe, self).__init__(
            pre_procs=ToChar() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensorList(),
            batch_procs=PackArrayBatch(),
        )
