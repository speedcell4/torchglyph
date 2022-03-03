import itertools
from collections import namedtuple, OrderedDict
from pathlib import Path
from typing import Iterable, Any, Type
from typing import Union, List, Tuple, NamedTuple, Dict

from torch.distributions.utils import lazy_property
from torch.utils.data import DataLoader as TorchDataLoader, SequentialSampler, RandomSampler
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from torchglyph.io import DownloadMixin
from torchglyph.pipe import Pipe
from torchglyph.sampler import BatchSampler

__all__ = [
    'Dataset',
    'DataLoader',
]


class Dataset(TorchDataset, DownloadMixin):
    def __init__(self, pipes: List[Dict[str, Pipe]], **kwargs) -> None:
        super(Dataset, self).__init__()

        self.pipes = {}
        self.names = []

        for ps in pipes:
            for name, pipe in ps.items():
                self.pipes[name] = pipe
                self.names.append(name)

        self.data = {}
        for datum, ps in zip(zip(*self.load(**kwargs)), pipes):
            for name, pipe in ps.items():
                self.data.setdefault(name, []).extend(datum)

    def get_size(self, item: Any) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {name: self.data[name][index] for name in self.names}

    def __len__(self) -> int:
        return len(next(iter(self.data.values())))

    @lazy_property
    def named_tuple(self) -> Type:
        return namedtuple(f'{self.__class__.__name__}Batch', field_names=self.names)

    @property
    def vocabs(self) -> NamedTuple:
        return self.named_tuple(**{
            name: pipe.vocab
            for name, pipe in self.pipes.items()
        })

    def collate_fn(self, batch: List[Dict[str, Any]]) -> NamedTuple:
        return self.named_tuple(**{
            name: pipe.collate_fn([data[name] for data in batch])
            for name, pipe in self.pipes.items()
        })

    @classmethod
    def load(cls, **kwargs) -> Iterable[Any]:
        raise NotImplementedError

    def dump(self, fp, batch: NamedTuple, prediction: Any, *args, **kwargs) -> None:
        raise NotImplementedError

    def state_dict(self, destination: OrderedDict = None, prefix: str = '',
                   keep_vars: bool = False) -> OrderedDict:
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        for name, datum in self.data.items():
            destination[prefix + name] = datum

        return destination

    def load_state_dict(self, state_dict: OrderedDict, strict: bool = True) -> None:
        names = set(self.names)
        for name, datum in state_dict.items():
            self.data[name] = datum
            if strict:
                names.remove(name)

        if strict:
            assert len(names) == 0

    def eval(self, path: Path, **kwargs):
        raise NotImplementedError

    @classmethod
    def new(cls, **kwargs) -> Tuple['DataLoader', ...]:
        raise NotImplementedError


class DataLoader(TorchDataLoader):
    dataset: Dataset

    @property
    def vocabs(self) -> NamedTuple:
        return self.dataset.vocabs

    @classmethod
    def new(cls, datasets: Tuple[Dataset, ...],
            batch_size: Union[int, Tuple[int, ...]],
            shuffle: bool = True, drop_last: bool = False) -> List['DataLoader']:
        assert len(datasets) > 0

        batch_sizes = batch_size
        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)

        with tqdm(desc='post-processing', total=sum(len(dataset.pipes) for dataset in datasets)) as progress:
            for index, dataset in enumerate(datasets):
                for name, pipe in dataset.pipes.items():
                    progress.set_postfix_str(f'{index}.{name}')
                    pipe.postprocess_(dataset)
                    progress.update(1)

        loaders = []

        for index, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes)):
            if index == 0 and shuffle:
                sampler = RandomSampler(dataset, generator=None)
            else:
                sampler = SequentialSampler(dataset)

            batch_sampler = BatchSampler(
                dataset=dataset, sampler=sampler, batch_size=batch_size,
                drop_last=index == 0 and drop_last,
            )

            loaders.append(DataLoader(
                dataset=dataset, batch_size=1,
                collate_fn=dataset.collate_fn,
                batch_sampler=batch_sampler,
            ))

        return loaders
