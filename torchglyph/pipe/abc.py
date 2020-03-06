from collections import Counter
from typing import Optional, Union, List, Any

from torchglyph.proc import Chain, Proc
from torchglyph.vocab import Vocab


class Pipeline(object):
    def __init__(self,
                 pre_procs: Optional[Union[Chain, Proc]],
                 vocab_procs: Optional[Union[Chain, Proc]],
                 post_procs: Optional[Union[Chain, Proc]],
                 batch_procs: Optional[Union[Chain, Proc]]) -> None:
        super(Pipeline, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self._pre_processing = Chain(pre_procs)
        self._vocab_processing = Chain(vocab_procs)
        self._post_processing = Chain(post_procs)
        self._batch_processing = Chain(batch_procs)

    def replace(self,
                pre_procs: Optional[Union[Chain, Proc]] = None,
                vocab_procs: Optional[Union[Chain, Proc]] = None,
                post_procs: Optional[Union[Chain, Proc]] = None,
                batch_procs: Optional[Union[Chain, Proc]] = None) -> 'Pipeline':
        return Pipeline(
            pre_procs=self._pre_processing if pre_procs is None else pre_procs,
            vocab_procs=self._vocab_processing if vocab_procs is None else vocab_procs,
            post_procs=self._post_processing if post_procs is None else post_procs,
            batch_procs=self._batch_processing if batch_procs is None else batch_procs,
        )

    def preprocess(self, *datasets) -> Counter:
        counter = Counter()
        for dataset in datasets:
            for key, pipe in dataset.pipelines.items():
                flag = f'@{key}_{self.preprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.instances[key] = [
                        self._pre_processing(ins, counter=counter)
                        for ins in dataset.instances[key]
                    ]
                    setattr(dataset, flag, True)

        return counter

    def postprocess(self, *datasets) -> 'Pipeline':
        _ = self.preprocess(*datasets)
        for dataset in datasets:
            for name, pipe in dataset.pipelines.items():
                flag = f'@{name}_{self.postprocess.__name__}_done'
                if pipe is self and not getattr(dataset, flag, False):
                    dataset.instances[name] = [
                        self._post_processing(ins, vocab=self.vocab)
                        for ins in dataset.instances[name]
                    ]
                    setattr(dataset, flag, True)

        return self

    def build_vocab(self, *datasets) -> 'Pipeline':
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
