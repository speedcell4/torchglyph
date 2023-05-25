import itertools
from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from datasets import DownloadConfig, DownloadManager
from torch.utils import data

from torchglyph import data_dir
from torchglyph.sampler import SortishBatchSampler, SortishDevSampler, SortishSampler

logger = getLogger(__name__)


class DataStore(object, metaclass=ABCMeta):
    name: str

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        raise NotImplementedError

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
        raise NotImplementedError

    @classmethod
    def get_tokenize_fn(cls, **kwargs):
        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            raise NotImplementedError

        return tokenize

    @classmethod
    def get_collate_fn(cls, **kwargs):
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            raise NotImplementedError

        return collate_fn

    @classmethod
    def predicate(cls, example) -> bool:
        raise NotImplementedError

    @classmethod
    def new(cls, **kwargs):
        raise NotImplementedError


class DataLoader(data.DataLoader):
    @classmethod
    def new(cls, data_stores: Tuple[DataStore, ...],
            collate_fn, batch_size: Union[int, Tuple[int, ...]],
            drop_last: bool = False, section_size: int = 1 << 12) -> List['DataLoader']:
        assert len(data_stores) > 0

        batch_sizes = batch_size
        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)

        loaders = []
        for index, (datastore, batch_size) in enumerate(zip(data_stores, batch_sizes)):
            sampler = SortishSampler(datastore, section_size=section_size)
            logger.debug(f'{index}.sampler => {sampler}')

            batch_sampler = (SortishBatchSampler if index == 0 else SortishDevSampler)(
                dataset=datastore, sampler=sampler, batch_size=batch_size,
                drop_last=index == 0 and drop_last,
            )
            logger.debug(f'{index}.batch_sampler => {batch_sampler}')

            loaders.append(DataLoader(
                dataset=datastore,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
            ))

        return loaders
