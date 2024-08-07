import itertools
from abc import ABCMeta
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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


class DataLoader(data.DataLoader):
    dataset: Dataset

    @classmethod
    def new(cls, data_sources: Tuple[Dataset, ...],
            collate_fn, batch_size: Union[int, Tuple[int, ...]],
            drop_last: bool = False, section_size: int = 1 << 12,
            sortish_key: str = 'size') -> List['DataLoader']:
        assert len(data_sources) > 0

        batch_sizes = batch_size
        if isinstance(batch_size, int):
            batch_sizes = itertools.repeat(batch_size)

        loaders = []
        for index, (data_source, batch_size) in enumerate(zip(data_sources, batch_sizes)):
            training = index == 0
            if not training:
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
