from torchglyph.dataset import Pipeline
from torchglyph.processes import ToChar, AddToCounter, BuildVocab, Numbering, ToTensorList, PackSubBatch


class PackedSubPipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSubPipe, self).__init__(
            pre_procs=ToChar() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensorList(),
            batch_procs=PackSubBatch(),
        )
