from typing import List, Iterator

from torch.utils.data import BatchSampler, Sampler

__all__ = [
    'SizedBatchSampler',
]


class SizedBatchSampler(BatchSampler):
    def __init__(self, dataset, sampler: Sampler[int], batch_size: int, drop_last: bool) -> None:
        super(SizedBatchSampler, self).__init__(sampler, batch_size, drop_last)

        self.dataset = dataset

        self.reset_indices()

    def reset_indices(self) -> None:
        self.indices, batch, batch_size = [], [], 0

        for index in self.sampler:
            data_size = self.dataset.get_size(self.dataset[index])
            batch.append(index)
            batch_size += data_size

            if batch_size >= self.batch_size:
                self.indices.append(batch)
                batch, batch_size = [], 0

        if not self.drop_last and batch_size > 0:
            self.indices.append(batch)

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.indices
        self.reset_indices()

    def __len__(self) -> int:
        num_batches = len(self.indices)

        if not self.drop_last:
            num_batches += self.batch_size - 1
        return num_batches // self.batch_size
