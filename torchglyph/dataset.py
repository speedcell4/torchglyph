import itertools
from collections import namedtuple, Counter
from typing import Union, List, Tuple, Dict, Any, Optional, NamedTuple, Callable

from torch.utils import data

from torchglyph.proc import Compose, PreProc, VocabProc, PostProc, BatchProc
from torchglyph.vocab import Vocab


class Pipeline(object):
    def __init__(self,
                 pre_procs: Optional[Union[Compose, PreProc]],
                 vocab_procs: Optional[Union[Compose, VocabProc]],
                 post_procs: Optional[Union[Compose, PostProc]],
                 batch_procs: Optional[Union[Compose, BatchProc]]) -> None:
        super(Pipeline, self).__init__()

        self.vocab: Optional[Union[Vocab]] = None

        self._pre_processing = Compose(pre_procs)
        self._vocab_processing = Compose(vocab_procs)
        self._post_processing = Compose(post_procs)
        self._batch_processing = Compose(batch_procs)

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
        self.vocab = self._vocab_processing(counter)
        return self

    def collate_fn(self, collected_ins: List[Any]) -> Any:
        return self._batch_processing(collected_ins, vocab=self.vocab)


class Dataset(data.Dataset):
    def __init__(self, instances: List[List[Any]], pipelines: List[Dict[str, Pipeline]]) -> None:
        super(Dataset, self).__init__()

        self.pipelines: Dict[str, Pipeline] = {
            key: pipe
            for pipes in pipelines
            for key, pipe in pipes.items()
        }
        self.Batch: Callable[[Any, ...], NamedTuple] = namedtuple('Batch', field_names=self.pipelines.keys())

        self.instances: Dict[str, List[Any]] = {}
        for ins, pipes in zip(zip(*instances), pipelines):
            for key, pipe in pipes.items():
                self.instances.setdefault(key, []).extend(ins)

        self._len = len(instances)

    def __getitem__(self, index: int) -> Dict[str, List[Any]]:
        return {
            key: ins[index]
            for key, ins in self.instances.items()
        }

    def __len__(self) -> int:
        return self._len

    def collact_fn(self, batch: List[NamedTuple]) -> NamedTuple:
        batch = self.Batch(*zip(*batch))
        return self.Batch(
            self.pipelines[key].collact_fn(collected_ins)
            for key, collected_ins in zip(batch._fields, batch)
        )


class DataLoader(data.DataLoader):
    @classmethod
    def iters(cls, datasets: Tuple[Dataset, ...], batch_size: Union[int, Tuple[int, ...]], shuffle: bool,
              num_workers: int = 1, pin_memory: bool = False, drop_last: bool = False) -> Tuple['DataLoader', ...]:
        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)
        else:
            batch_sizes = batch_size

        for dataset in datasets:
            for pipe in dataset.pipelines.values():
                pipe.postprocess(dataset)

        return tuple(
            DataLoader(
                dataset=dataset, shuffle=shuffle and (index == 0),
                batch_size=batch_size, collate_fn=dataset.collact_fn,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            )
            for index, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes))
        )
