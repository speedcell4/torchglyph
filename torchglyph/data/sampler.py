from logging import getLogger
from typing import Iterator, List

import torch
from datasets import Dataset
from torch.utils import data

logger = getLogger(__name__)


class SequentialSampler(data.SequentialSampler):
    def __init__(self, data_source: Dataset) -> None:
        super(SequentialSampler, self).__init__(data_source=data_source)


class RandomSampler(data.RandomSampler):
    def __init__(self, data_source: Dataset, replacement: bool = False) -> None:
        super(RandomSampler, self).__init__(
            data_source=data_source, replacement=replacement,
            num_samples=None, generator=torch.default_generator,
        )


class SortishSampler(data.Sampler[int]):
    def __init__(self, data_source: Dataset, section: int = 1 << 12, sortish_key: str = 'token_size') -> None:
        super(SortishSampler, self).__init__()

        self.section = section
        self.descending = True

        self.sizes = data_source[sortish_key]
        self.indices = []

    def __len__(self) -> int:
        return len(self.sizes)

    def extend(self, indices: List[int]):
        self.indices.extend(indices)

    def __iter__(self) -> Iterator[int]:

        sizes = torch.tensor(self.sizes, dtype=torch.long)
        indices = torch.randperm(len(self), dtype=torch.long, generator=torch.default_generator)

        if len(self.indices) > 0:
            indices = torch.cat([torch.tensor(self.indices, dtype=torch.long), indices], dim=0)
            self.indices = []

        for index in torch.split(indices, self.section, dim=0):
            args = torch.argsort(sizes[index], dim=0, descending=self.descending)
            self.descending ^= True

            yield from index[args].detach().cpu().tolist()


class SortishBatchSampler(data.BatchSampler):
    sampler: SortishSampler

    def __init__(self, sampler: SortishSampler, batch_size: int, training: bool, drop_last: bool = False) -> None:
        super(SortishBatchSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        self.sizes = sampler.sizes
        self.training = training

    def __iter__(self) -> Iterator[List[int]]:
        batch, batch_size = [], 0

        while True:
            for index in self.sampler:
                if not (0 < self.sizes[index] <= self.batch_size):
                    logger.warning(f'sizes[{index}] = {self.sizes[index]} is not in (0, {self.batch_size}]')
                    continue

                if batch_size + self.sizes[index] > self.batch_size:
                    yield batch
                    batch, batch_size = [], 0

                batch.append(index)
                batch_size += self.sizes[index]

            if len(batch) > 0 and not self.drop_last:
                if self.training:
                    self.sampler.extend(batch)
                else:
                    yield batch

            batch, batch_size = [], 0
            if not self.training:
                break
