import itertools
from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

from datasets import Dataset, DownloadConfig, DownloadManager
from torch.utils import data

from torchglyph import data_dir
from torchglyph.data.sampler import SortishBatchSampler, SortishSampler
from torchglyph.dist import get_rank, get_world_size

logger = getLogger(__name__)


class Archive(object, metaclass=ABCMeta):
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


def unpack(obj: Union[Any, Tuple[Any, ...]]):
    if not isinstance(obj, (list, tuple)):
        return itertools.repeat(obj)

    return obj


class DataLoader(data.DataLoader):
    dataset: Dataset

    @classmethod
    def new(cls, data_sources: Tuple[Dataset, ...],
            collate_fn: Union[Callable, Tuple[Callable, ...]],
            batch_size: Union[int, Tuple[int, ...]],
            sharding: Union[bool, Tuple[bool, ...]] = True,
            drop_last: Union[bool, Tuple[bool, ...]] = False,
            section_size: Union[int, Tuple[int, ...]] = 1 << 12,
            sortish_key: Union[str, Tuple[str, ...]] = 'size') -> List['DataLoader']:
        assert len(data_sources) > 0

        loaders = []
        for index, (data_source, collate_fn, batch_size, sharding, drop_last, section_size, sortish_key) in enumerate(zip(
                data_sources,
                unpack(collate_fn),
                unpack(batch_size),
                unpack(sharding),
                unpack(drop_last),
                unpack(section_size),
                unpack(sortish_key),
        )):
            training = index == 0
            if sharding:
                data_source = data_source.select(range(get_rank(), len(data_source), get_world_size()))

            sampler = SortishSampler(
                data_source=data_source,
                section=section_size,
                sortish_key=sortish_key,
            )
            logger.debug(f'{index}.sampler => {sampler}')

            batch_sampler = SortishBatchSampler(
                sampler=sampler, batch_size=batch_size,
                training=training, drop_last=training and drop_last,
            )
            logger.debug(f'{index}.batch_sampler => {batch_sampler}')

            loaders.append(DataLoader(
                dataset=data_source,
                collate_fn=collate_fn,
                batch_sampler=batch_sampler,
            ))

        return loaders
