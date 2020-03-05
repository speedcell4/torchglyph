from torchglyph.dataset import Pipeline
from torchglyph.processes import ToSub, AddToCounter, BuildVocab, Numbering, ToTensorList, PackSubBatch, PadSubBatch, \
    Union, ToPad


class PaddedSubPipe(Pipeline):
    def __init__(self, pad_token: Union[int, str] = '<pad>', batch_first: bool = True) -> None:
        super(PaddedSubPipe, self).__init__(
            pre_procs=ToSub() + AddToCounter(),
            vocab_procs=BuildVocab(pad_token=pad_token),
            post_procs=Numbering() + ToTensorList() + ToPad(pad_token=pad_token, batch_first=True),
            batch_procs=PadSubBatch(pad_token=pad_token, batch_first=batch_first),
        )


class PackedSubPipe(Pipeline):
    def __init__(self) -> None:
        super(PackedSubPipe, self).__init__(
            pre_procs=ToSub() + AddToCounter(),
            vocab_procs=BuildVocab(),
            post_procs=Numbering() + ToTensorList(),
            batch_procs=PackSubBatch(),
        )
