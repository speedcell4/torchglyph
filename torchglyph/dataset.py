import itertools
import uuid
from collections import namedtuple, Counter
from typing import Union, List, Tuple, Dict, Any, Optional, NamedTuple, Callable

from torch.utils import data

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


class Dataset(data.Dataset):
    def __init__(self, instances: List[List[Any]], pipelines: List[Dict[str, Pipeline]]) -> None:
        super(Dataset, self).__init__()

        self.pipelines: Dict[str, Pipeline] = {
            key: pipe
            for pipes in pipelines
            for key, pipe in pipes.items()
        }
        self.Batch: Callable[[Any, ...], NamedTuple] = namedtuple(
            f'Batch_{str(uuid.uuid4())[:8]}', field_names=self.pipelines.keys())
        if self.Batch.__name__ not in globals():
            globals()[self.Batch.__name__] = self.Batch

        self.instances: Dict[str, List[Any]] = {}
        for ins, pipes in zip(zip(*instances), pipelines):
            for key, pipe in pipes.items():
                self.instances.setdefault(key, []).extend(ins)

        self._len = len(instances)

    def __getitem__(self, index: int) -> NamedTuple:
        return self.Batch(*[
            self.instances[key][index] for key in self.Batch._fields
        ])

    def __len__(self) -> int:
        return self._len

    def collate_fn(self, batch: List[NamedTuple]) -> NamedTuple:
        batch = self.Batch(*zip(*batch))
        return self.Batch(*[
            self.pipelines[key].collate_fn(collected_ins)
            for key, collected_ins in zip(batch._fields, batch)
        ])

    @classmethod
    def loaders(cls, *args, **kwargs) -> Tuple['DataLoader', ...]:
        raise NotImplementedError


class DataLoader(data.DataLoader):
    @classmethod
    def loaders(cls, datasets: Tuple[Dataset, ...], batch_size: Union[int, Tuple[int, ...]], shuffle: bool,
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
                batch_size=batch_size, collate_fn=dataset.collate_fn,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            )
            for index, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes))
        )
