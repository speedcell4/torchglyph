import warnings
from typing import Iterator, List

import torch
from datasets import Dataset
from torch.utils import data as utils

from torchglyph.dist import get_rank


class RandomSampler(utils.RandomSampler):
    def __init__(self, dataset: Dataset, replacement: bool = False, num_samples: int = None) -> None:
        super(RandomSampler, self).__init__(data_source=dataset, replacement=replacement, num_samples=num_samples)

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            self.generator = torch.default_generator

        yield from super(RandomSampler, self).__iter__()


class SequentialSampler(utils.SequentialSampler):
    def __init__(self, dataset: Dataset) -> None:
        super(SequentialSampler, self).__init__(data_source=dataset)


class SortishSampler(utils.Sampler[int]):
    def __init__(self, dataset: Dataset, section_size: int) -> None:
        super(SortishSampler, self).__init__(data_source=dataset)

        self.section_size = section_size
        self.sizes = dataset['size']

        self.last_indices = []
        self.descending = None

    def __len__(self) -> int:
        return len(self.sizes)

    def extend(self, last_indices: List[int]) -> None:
        self.last_indices.extend(last_indices)

    def __iter__(self) -> Iterator[int]:
        if self.descending is None:
            self.descending = get_rank() % 2 == 0

        sizes = torch.tensor(self.sizes, dtype=torch.long)
        randperm = torch.randperm(len(self), dtype=torch.long, generator=torch.default_generator)

        if len(self.last_indices) > 0:
            last_indices = torch.tensor(self.last_indices, dtype=torch.long)
            self.last_indices = []

            randperm = torch.cat([last_indices, randperm], dim=0)

        for index in torch.split(randperm, self.section_size, dim=0):
            args = torch.argsort(sizes[index], dim=0, descending=self.descending)
            self.descending ^= True

            yield from index[args].detach().cpu().tolist()


class SortishBatchSampler(utils.BatchSampler):
    def __init__(self, dataset: Dataset, sampler: SortishSampler, batch_size: int, drop_last: bool = False) -> None:
        super(SortishBatchSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        self.sizes = dataset['size']

    def __iter__(self) -> Iterator[List[int]]:
        batch, current_size = [], 0

        while True:
            for index in self.sampler:
                if self.sizes[index] + current_size > self.batch_size:
                    if current_size > 0:
                        yield batch
                        batch, current_size = [], 0
                    else:
                        warnings.warn(
                            f'example {index} :: {self.sizes[index]} is dropped, '
                            f'consider increasing batch size ({current_size})?',
                        )
                        continue

                batch.append(index)
                current_size += self.sizes[index]

            if not self.drop_last:
                self.sampler.extend(batch)
            batch, current_size = [], 0


class SortishDevSampler(utils.BatchSampler):
    def __init__(self, dataset: Dataset, sampler: SortishSampler, batch_size: int, drop_last: bool = False) -> None:
        super(SortishDevSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        self.sizes = dataset['size']

    def __iter__(self) -> Iterator[List[int]]:
        batch, current_size = [], 0

        for index in self.sampler:
            if self.sizes[index] + current_size > self.batch_size:
                if current_size > 0:
                    yield batch
                    batch, current_size = [], 0
                else:
                    warnings.warn(
                        f'example {index} :: {self.sizes[index]} is dropped, '
                        f'consider increasing batch size ({current_size})?',
                    )
                    continue

            batch.append(index)
            current_size += self.sizes[index]

        if not self.drop_last and len(batch) > 0:
            yield batch
