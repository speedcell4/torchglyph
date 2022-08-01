import warnings
from typing import Iterator
from typing import List

import torch
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data import RandomSampler as _RandomSampler
from torch.utils.data import SequentialSampler as _SequentialSampler


class RandomSampler(_RandomSampler):
    def __call__(self, *args, **kwargs) -> 'RandomSampler':
        return self


class SequentialSampler(_SequentialSampler):
    def __call__(self, *args, **kwargs) -> 'SequentialSampler':
        return self


class SortishSampler(Sampler[int]):
    def __init__(self, data_source, section_size: int) -> None:
        super(SortishSampler, self).__init__(data_source=data_source)

        self.section_size = section_size
        self.sizes = torch.tensor(data_source.sizes, dtype=torch.long, device=torch.device('cpu'))

    def __len__(self) -> int:
        return self.sizes.size()[0]

    def __call__(self, last_indices: List[int], *args, **kwargs) -> 'SortishSampler':
        self.last_indices = last_indices
        return self

    def __iter__(self) -> Iterator[int]:
        randperm = torch.randperm(len(self), dtype=torch.long, device=torch.device('cpu'))
        if len(getattr(self, 'last_indices', [])) > 0:
            last_indices = torch.tensor(self.last_indices, dtype=torch.long, device=torch.device('cpu'))
            randperm = torch.cat([last_indices, randperm], dim=0)

        for index, indices in enumerate(torch.split(randperm, split_size_or_sections=self.section_size, dim=0)):
            sorting = torch.argsort(self.sizes[indices], dim=0, descending=index % 2 == 0)
            yield from indices[sorting].detach().cpu().tolist()


class SizedBatchSampler(BatchSampler):
    def __init__(self, data_source, sampler: SortishSampler, batch_size: int) -> None:
        super(SizedBatchSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=False)

        self.sizes = data_source.sizes

    def __iter__(self) -> Iterator[List[int]]:
        batch, batch_size = [], 0

        while True:
            for index in self.sampler(batch):
                if self.sizes[index] + batch_size > self.batch_size:
                    if batch_size > 0:
                        yield batch
                        batch, batch_size = [], 0
                    else:
                        warnings.warn(
                            f'example {index} :: {self.sizes[index]} is dropped, '
                            f'consider increasing batch size ({batch_size})?',
                        )

                batch.append(index)
                batch_size += self.sizes[index]

            if not isinstance(self.sampler, SortishSampler) and len(batch) > 0:
                yield batch
                break
