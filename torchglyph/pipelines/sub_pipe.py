from torchglyph.dataset import Pipeline
from torchglyph.processes import ToChar, AddToCounter, BuildVocab, Numbering, ToTensorList, PackSubBatch, PadSubBatch, \
    Union


class PaddedSubPipe(Pipeline):
    def __init__(self, pad_token: Union[int, str], batch_first: bool = True) -> None:
        super(PaddedSubPipe, self).__init__(
            pre_procs=ToChar() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensorList(),
            batch_procs=PadSubBatch(pad_token=pad_token, batch_first=batch_first),
        )


class PackedSubPipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSubPipe, self).__init__(
            pre_procs=ToChar() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensorList(),
            batch_procs=PackSubBatch(),
        )
