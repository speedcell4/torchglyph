from typing import List, Iterator

from torch.utils.data import BatchSampler as _BatchSampler, Sampler

__all__ = [
    'BatchSampler',
]


class BatchSampler(_BatchSampler):
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
