import itertools
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Any
from typing import Union, List, Type, Tuple, NamedTuple, Dict

from torch.utils import data

from torchglyph import data_path
from torchglyph.io import download_and_unzip
from torchglyph.pipe import Pipe


class Dataset(data.Dataset):
    urls: List[Union[Tuple[str, ...]]]

    def __init__(self, pipes: List[Dict[str, Pipe]], **kwargs) -> None:
        super(Dataset, self).__init__()

        self.pipes: Dict[str, Pipe] = {
            key: pipe
            for ps in pipes
            for key, pipe in ps.items()
        }
        self.Batch: Type[NamedTuple] = namedtuple(
            f'Batch_{str(uuid.uuid4())[:8]}', field_names=self.pipes.keys())
        if self.Batch.__name__ not in globals():
            globals()[self.Batch.__name__] = self.Batch

        self.data: Dict[str, List[Any]] = {}
        for ins, pipes in zip(zip(*self.iter(**kwargs)), pipes):
            for key, pipe in pipes.items():
                self.data.setdefault(key, []).extend(ins)

        self._len = 0  # cache the number of instances
        for datum in self.data.values():
            self._len = len(datum)
            break

    def __getitem__(self, index: int) -> NamedTuple:
        return self.Batch(*[
            self.data[key][index]
            for key in self.Batch._fields
        ])

    def __len__(self) -> int:
        return self._len

    @property
    def vocabs(self) -> NamedTuple:
        return self.Batch(**{
            key: pipe.vocab
            for key, pipe in self.pipes.items()
        })

    def collate_fn(self, batch: List[NamedTuple]) -> NamedTuple:
        batch = self.Batch(*zip(*batch))
        return self.Batch(*[
            self.pipes[key].collate_fn(collected_ins)
            for key, collected_ins in zip(batch._fields, batch)
        ])

    @classmethod
    def paths(cls, root: Path = data_path) -> Tuple[Path, ...]:
        ans = []
        for url, name, *filenames in cls.urls:
            if len(filenames) == 0:
                filenames = [name]
            if any(not (root / cls.__name__.lower() / n).exists() for n in filenames):
                download_and_unzip(url, root / cls.__name__.lower() / name)
            for n in filenames:
                ans.append(root / cls.__name__.lower() / n)

        return tuple(ans)

    @classmethod
    def iter(cls, **kwargs) -> Iterable[List[Any]]:
        raise NotImplementedError

    def dump(self, fp, batch: NamedTuple, *args, **kwargs) -> None:
        raise NotImplementedError

    @classmethod
    def new(cls, *args, **kwargs) -> Tuple['DataLoader', ...]:
        raise NotImplementedError


class DataLoader(data.DataLoader):
    @property
    def vocabs(self) -> NamedTuple:
        return self.dataset.vocabs

    @classmethod
    def new(cls, datasets: Tuple[Dataset, ...],
            batch_size: Union[int, Tuple[int, ...]], shuffle: bool,
            num_workers: int = 0, pin_memory: bool = False,
            drop_last: bool = False) -> Tuple['DataLoader', ...]:
        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)
        else:
            batch_sizes = batch_size

        for dataset in datasets:
            for pipe in dataset.pipes.values():
                pipe.postprocess(dataset)

        return tuple(
            DataLoader(
                dataset=dataset, shuffle=shuffle and (index == 0),
                batch_size=batch_size, collate_fn=dataset.collate_fn,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            )
            for index, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes))
        )
