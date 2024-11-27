from logging import getLogger

from datasets import Dataset
from torch import distributed
from torch.utils import data

logger = getLogger(__name__)


class _SortishSampler(data.Sampler[int]):
    def __init__(self, ds: Dataset, key: str, section_size: int, sharding: bool) -> None:
        super(_SortishSampler, self).__init__()

        self.section_size = section_size

        ds = ds.select_columns([key]).rename_column(key, 'key')
        ds = ds.add_column('idx', range(len(ds)))
        if sharding and distributed.is_initialized():
            ds = ds.shard(
                num_shards=distributed.get_world_size(),
                index=distributed.get_rank(),
            )

        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)


class SequentialSortishSampler(_SortishSampler):
    def __init__(self, ds: Dataset, key: str, section_size: int, sharding: bool) -> None:
        super(SequentialSortishSampler, self).__init__(ds=ds, key=key, section_size=section_size, sharding=sharding)
        self.ds = self.ds.sort(column_names=['key'], reverse=True)

    def __iter__(self):
        for batch in self.ds.iter(batch_size=self.section_size, drop_last_batch=False):
            yield list(zip(batch['idx'], batch['key']))


class RandomSortishSampler(_SortishSampler):
    def __iter__(self):
        idx, key, reverse = [], [], True

        while True:
            for batch in self.ds.shuffle().iter(batch_size=self.section_size, drop_last_batch=False):
                idx.extend(batch['idx'])
                key.extend(batch['key'])

                if len(idx) < self.section_size:
                    continue

                yield sorted(list(zip(idx, key)), key=lambda item: item[1], reverse=reverse)
                idx, key, reverse = [], [], not reverse


class SortishBatchSampler(data.BatchSampler):
    def __init__(self, sampler, batch_size: int, drop_last: bool = False) -> None:
        super(SortishBatchSampler, self).__init__(sampler=sampler, batch_size=batch_size, drop_last=drop_last)

    def __iter__(self):
        batch, size = [], 0

        for examples in self.sampler:
            for idx, key in examples:
                if size + key > self.batch_size:
                    yield batch
                    batch, size = [], 0

                batch.append(idx)
                size += key

        if len(batch) > 0 and not self.drop_last:
            yield batch
