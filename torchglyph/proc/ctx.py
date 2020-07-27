from typing import List

from allennlp.data import Instance as AllenInstance, Vocabulary as AllenVocabulary
from allennlp.data.dataset import Batch as AllenBatch
from torch import Tensor

from torchglyph.io import toggle_loggers
from torchglyph.proc import Proc

toggle_loggers('allennlp', False)


class PadELMo(Proc):
    def __call__(self, data: List[AllenInstance], *args, **kwargs) -> Tensor:
        batch = AllenBatch(data)
        batch.index_instances(AllenVocabulary())
        return batch.as_tensor_dict()['elmo']['character_ids']
