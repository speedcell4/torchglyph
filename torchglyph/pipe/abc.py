from collections import Counter
from typing import Optional, Union, List, Any

from torchglyph.proc import Proc, PaLP, compress
from torchglyph.vocab import Vocab


class Pipe(object):
    def __init__(self, pre_procs: PaLP = None, vocab_procs: PaLP = None,
                 post_procs: PaLP = None, batch_procs: PaLP = None) -> None:
        super(Pipe, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self._pre_processing = Proc.from_list(compress(pre_procs))
        self._vocab_processing = Proc.from_list(compress(vocab_procs))
        self._post_processing = Proc.from_list(compress(post_procs))
        self._batch_processing = Proc.from_list(compress(batch_procs))

    def preprocess(self, *datasets) -> Counter:
        counter = Counter()
        for dataset in datasets:
            for key, pipe in dataset.pipes.items():
                flag = f'@{key}_{self.preprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.data[key] = [
                        self._pre_processing(ins, counter=counter)
                        for ins in dataset.data[key]
                    ]
                    setattr(dataset, flag, True)

        return counter

    def postprocess(self, *datasets) -> 'Pipe':
        _ = self.preprocess(*datasets)
        for dataset in datasets:
            for name, pipe in dataset.pipes.items():
                flag = f'@{name}_{self.postprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.data[name] = [
                        self._post_processing(ins, vocab=self.vocab)
                        for ins in dataset.data[name]
                    ]
                    setattr(dataset, flag, True)

        return self

    def build_vocab(self, *datasets) -> 'Pipe':
        counter = self.preprocess(*datasets)
        vocab = self._vocab_processing(counter)
        if isinstance(vocab, Vocab):
            self.vocab = vocab
        else:
            raise ValueError(f"vocabulary building produced '{type(vocab).__name__}', "
                             f"instead of '{Vocab.__name__}'")
        return self

    def collate_fn(self, collected_ins: List[Any]) -> Any:
        return self._batch_processing(collected_ins, vocab=self.vocab)
