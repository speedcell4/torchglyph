import itertools
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Any, Type
from typing import Union, List, Tuple, NamedTuple, Dict

from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from torchglyph.io import DownloadMixin
from torchglyph.pipe import Pipe

__all__ = [
    'Dataset',
    'DataLoader',
]


class Dataset(TorchDataset, DownloadMixin):
    def __init__(self, pipes: List[Dict[str, Pipe]], **kwargs) -> None:
        super(Dataset, self).__init__()

        self.pipes: Dict[str, Pipe] = {
            name: pipe
            for ps in pipes
            for name, pipe in ps.items()
        }
        self.Batch: Type = namedtuple(
            typename=f'Batch_{str(uuid.uuid4())[:8]}',
            field_names=list(self.pipes.keys()),
        )
        if self.Batch.__name__ not in globals():
            globals()[self.Batch.__name__] = self.Batch

        self.data: Dict[str, List[Any]] = {}

        for datum, ps in zip(zip(*self.load(**kwargs)), pipes):
            for name, pipe in ps.items():
                self.data.setdefault(name, []).extend(datum)

    def transpose(self) -> None:
        names, data = zip(*self.data.items())
        names, data = list(names), zip(*data)
        self.data = [self.Batch(**dict(zip(names, datum))) for datum in data]

    def __getitem__(self, index: int) -> NamedTuple:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    @property
    def vocabs(self) -> NamedTuple:
        return self.Batch(**{
            name: pipe.vocab
            for name, pipe in self.pipes.items()
        })

    def collate_fn(self, batch: List[NamedTuple]) -> NamedTuple:
        data = self.Batch(*zip(*batch))
        return self.Batch(*[
            self.pipes[name].collate_fn(datum)
            for name, datum in zip(self.Batch._fields, data)
        ])

    @classmethod
    def load(cls, **kwargs) -> Iterable[Any]:
        raise NotImplementedError

    def dump(self, fp, batch: NamedTuple, prediction: Any, *args, **kwargs) -> None:
        raise NotImplementedError

    def eval(self, path: Path, **kwargs):
        raise NotImplementedError

    def viz(self, path: Path, **kwargs):
        raise NotImplementedError

    @classmethod
    def new(cls, **kwargs) -> Tuple['DataLoader', ...]:
        raise NotImplementedError


class DataLoader(TorchDataLoader):
    @property
    def vocabs(self) -> NamedTuple:
        return self.dataset.vocabs

    @classmethod
    def new(cls, datasets: Tuple[Dataset, ...],
            batch_size: Union[int, Tuple[int, ...]],
            shuffle: bool = True, drop_last: bool = False) -> Tuple['DataLoader', ...]:
        assert len(datasets) > 0

        batch_sizes = batch_size
        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)

        iteration = tqdm(
            desc='processing datasets',
            total=len(datasets) * (len(datasets[0].pipes) + 1),
        )
        for dataset in datasets:
            for name, pipe in dataset.pipes.items():
                pipe.postprocess(dataset, name=name)
                iteration.update(1)
                iteration.set_postfix_str(f'{name}')
            dataset.transpose()
            iteration.update(1)
            iteration.set_postfix_str('transpose')
        iteration.close()

        return tuple(
            DataLoader(
                dataset=dataset, batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle if index == 0 else False,
                drop_last=drop_last if index == 0 else False,
            )
            for index, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes))
        )
