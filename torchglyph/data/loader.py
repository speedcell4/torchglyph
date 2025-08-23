from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import List, Tuple

from datasets import Dataset, DownloadConfig, DownloadManager
from torch.utils import data

from torchglyph import data_dir
from torchglyph.data.sampler import RandomSortishSampler, SequentialSortishSampler, SortishBatchSampler

logger = getLogger(__name__)


class Archive(object, metaclass=ABCMeta):
    name: str

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        raise NotImplementedError()

    @classmethod
    def paths(cls, root: Path = data_dir, **kwargs) -> List[Path]:
        out = []

        dataset_name = getattr(cls, 'name', cls.__name__).lower()
        for url, *names in cls.urls(**kwargs):
            Download_manager = DownloadManager(
                dataset_name=dataset_name,
                download_config=DownloadConfig(
                    cache_dir=root / dataset_name,
                    download_desc=f'Downloading {url}',
                ),
            )

            archive = Path(Download_manager.download_and_extract(url))
            out.append(archive)

        return out

    @classmethod
    def load(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    def new(cls, **kwargs):
        raise NotImplementedError()


class DataLoader(data.DataLoader):
    @classmethod
    def new_train(cls, *datasets: Dataset, collate_fn,
                  batch_size: int, key: str, section_size: int, sharding: bool = False):
        return tuple(cls(
            dataset=ds,
            collate_fn=collate_fn,
            batch_sampler=SortishBatchSampler(
                sampler=RandomSortishSampler(ds=ds, key=key, section_size=section_size, sharding=sharding),
                batch_size=batch_size,
                drop_last=False,
            ),
        ) for ds in datasets)

    @classmethod
    def new_eval(cls, *datasets: Dataset, collate_fn,
                 batch_size: int, key: str, section_size: int, sharding: bool = True):
        return tuple(cls(
            dataset=ds,
            collate_fn=collate_fn,
            batch_sampler=SortishBatchSampler(
                sampler=SequentialSortishSampler(ds=ds, key=key, section_size=section_size, sharding=sharding),
                batch_size=batch_size,
                drop_last=False,
            ),
        ) for ds in datasets)
