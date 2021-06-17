from typing import List

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import pack_sequence

from torchglyph.proc import Proc


class PackList(Proc):
    def __init__(self) -> None:
        super(PackList, self).__init__()

    def __call__(self, sequences: List[Tensor], **kwargs) -> PackedSequence:
        return pack_sequence(sequences)
