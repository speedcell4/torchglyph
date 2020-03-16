import itertools
import uuid
from collections import namedtuple
from pathlib import Path
from typing import Iterable, Any, TextIO
from typing import Union, List, Type, Tuple, NamedTuple, Dict

from torch.utils import data
from tqdm import tqdm

from torchglyph import data_path
from torchglyph.io import download_and_unzip
from torchglyph.pipe import Pipe


class Dataset(data.Dataset):
    urls: List[Union[Tuple[str, ...]]]

    def __init__(self, pipes: List[Dict[str, Pipe]], **load_kwargs) -> None:
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
        for ins, pipes in zip(zip(*self.load(**load_kwargs)), pipes):
            for key, pipe in pipes.items():
                self.data.setdefault(key, []).extend(ins)

    def _transpose(self) -> None:
        keys, values = zip(*self.data.items())
        keys = list(keys)
        values = zip(*values)
        self.data = [self.Batch(**dict(zip(keys, ins))) for ins in values]

    def __getitem__(self, index: int) -> NamedTuple:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

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
    def load(cls, **kwargs) -> Iterable[Any]:
        raise NotImplementedError

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[Any], *args, **kwargs) -> None:
        raise NotImplementedError

    def eval(self, path: Path):
        raise NotImplementedError

    def viz(self, path: Path):
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
        assert len(datasets) > 0

        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)
        else:
            batch_sizes = batch_size

        iteration = tqdm(
            desc='post-processing datasets',
            total=len(datasets) * (len(datasets[0].pipes) + 1),
        )
        for dataset in datasets:
            for key, pipe in dataset.pipes.items():
                pipe.postprocess(dataset)
                iteration.update(1)
                iteration.set_postfix_str(f'{key}')
            dataset._transpose()
            iteration.update(1)
            iteration.set_postfix_str('transpose')
        iteration.close()

        return tuple(
            DataLoader(
                dataset=dataset, shuffle=shuffle and (index == 0),
                batch_size=batch_size, collate_fn=dataset.collate_fn,
                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
            )
            for index, (dataset, batch_size) in enumerate(zip(datasets, batch_sizes))
        )
