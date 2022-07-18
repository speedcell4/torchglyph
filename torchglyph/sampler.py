import warnings
from typing import Iterator
from typing import List

import torch
from torch.utils.data import BatchSampler, Sampler


class SortishSampler(Sampler[int]):
    def __init__(self, data_source, chunk_size: int) -> None:
        super(SortishSampler, self).__init__(data_source=data_source)

        self.chunk_size = chunk_size
        self.sizes = data_source.sizes

    def __len__(self) -> int:
        return len(self.sizes)

    def __iter__(self) -> Iterator[int]:
        sizes = torch.tensor(self.sizes, dtype=torch.long, device=torch.device('cpu'))
        permutation = torch.randperm(len(self.sizes), dtype=torch.long, device=torch.device('cpu'))

        chunks = [
            permutation[index:index + self.chunk_size]
            for index in range(0, len(self.sizes), self.chunk_size)
        ]
        for k, chunk in enumerate(chunks):
            indices = torch.argsort(sizes[chunk], dim=0, descending=k % 2 == 1)
            yield from chunk[indices].detach().tolist()


class SortishBatchSampler(BatchSampler):
    def __init__(self, data_source, sampler: Sampler[int],
                 batch_size: int, drop_last: bool) -> None:
        super(SortishBatchSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

        self.sizes = data_source.sizes

    def __len__(self) -> int:
        return len(self.sizes)

    def __iter__(self) -> Iterator[List[int]]:
        batch, batch_size = [], 0

        for index in self.sampler:
            if self.sizes[index] + batch_size > self.batch_size:
                if batch_size > 0:
                    yield batch
                    batch, batch_size = [], 0
                else:
                    warnings.warn(f'example {index} is dropped, consider increasing batch size?')

            batch.append(index)
            batch_size += self.sizes[index]

        if not self.drop_last and len(batch) > 0:
            yield batch
