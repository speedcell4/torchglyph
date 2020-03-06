from torchglyph.dataset import Pipeline
from torchglyph.proc import ToLength, PadTokBatch


class SeqLengthPipe(Pipeline):
    def __init__(self) -> None:
        super(SeqLengthPipe, self).__init__(
            pre_procs=None,
            vocab_procs=None,
            post_procs=ToLength(),
            batch_procs=PadTokBatch(),
        )
