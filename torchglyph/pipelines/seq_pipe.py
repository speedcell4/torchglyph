from torchglyph.dataset import Pipeline
from torchglyph.processes import UpdateCounter, BuildVocab, Numbering, ToTensor, PackBatch


class SeqPipe(Pipeline):
    def __init__(self) -> None:
        super(SeqPipe, self).__init__(
            pre_procs=UpdateCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensor(),
            batch_procs=PackBatch(),
        )
