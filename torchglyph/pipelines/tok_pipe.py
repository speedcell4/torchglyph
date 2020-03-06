from torchglyph.dataset import Pipeline
from torchglyph.proc import ToLength, ToTensor, AddToCounter, BuildVocab, Numbering


class PaddedTokPipe(Pipeline):
    def __init__(self) -> None:
        super(PaddedTokPipe, self).__init__(
            pre_procs=AddToCounter(),
            vocab_procs=BuildVocab(pad_token=None),
            post_procs=Numbering(),
            batch_procs=ToTensor(),
        )


class SeqLengthPipe(Pipeline):
    def __init__(self) -> None:
        super(SeqLengthPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToLength(),
            batch_procs=ToTensor(),
        )
