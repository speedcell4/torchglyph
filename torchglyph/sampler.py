from typing import Iterator
from typing import List

import torch
from torch.utils.data import BatchSampler as TorchBatchSampler, Sampler

from torchglyph.dataset import DatasetABC

__all__ = [
    'BatchSampler',
]


class SortishSampler(Sampler[int]):
    def __init__(self, data_source: DatasetABC, chunk_size: int) -> None:
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


class BatchSampler(TorchBatchSampler):
    def __init__(self, dataset, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        super(BatchSampler, self).__init__(sampler, batch_size, drop_last)

        self.dataset = dataset
        self.reset_indices()

    def reset_indices(self) -> None:
        self.batch_indices = []

        batch_size = 0
        batch_indices = []
        for index in self.sampler:
            data_size = self.dataset.get_size(self.dataset[index])

            if (data_size + batch_size) > self.batch_size:
                if batch_size > 0:
                    self.batch_indices.append(batch_indices)
                    batch_size = 0
                    batch_indices = []
                else:
                    continue

            batch_indices.append(index)
            batch_size += data_size

        if not self.drop_last and batch_size > 0:
            self.batch_indices.append(batch_indices)

    def __len__(self) -> int:
        return len(self.batch_indices)

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.batch_indices
        self.reset_indices()
