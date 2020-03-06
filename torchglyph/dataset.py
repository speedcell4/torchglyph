import itertools
import uuid
from collections import namedtuple
from typing import Union, List, Tuple, Dict, Any, NamedTuple, Callable

from torch.utils import data

from torchglyph.pipe import Pipe


class Dataset(data.Dataset):
    def __init__(self, instances: List[List[Any]], pipelines: List[Dict[str, Pipe]]) -> None:
        super(Dataset, self).__init__()

        self.pipelines: Dict[str, Pipe] = {
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
