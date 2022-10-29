from logging import getLogger
from typing import List, Type, Any

import torch
from datasets import load_dataset
from filelock import FileLock
from tqdm import tqdm

from benchmark.generator import device
from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.nn.plm import PLM
from torchglyph.nn.plm.mono import BartLarge
from torchglyph.pipe import CattedNumListPipe

logger = getLogger(__name__)


class TokenPipe(CattedNumListPipe):
    def __init__(self):
        super(TokenPipe, self).__init__(
            device=device,
            dtype=torch.long,
        )


class AbstractSummarization(Dataset):
    lang: str
    subset: str = None

    src_key: str
    tgt_key: str

    src_max_length: int = 512
    tgt_max_length: int = 128

    train_max_size: int = None
    dev_max_size: int = None
    test_max_size: int = None

    @classmethod
    def load(cls, split: str, plm: PLM, max_size: int = None, **kwargs):
        cache = data_dir / cls.name / plm.pretrained_model_name
        if cls.subset is not None:
            cache = cache / cls.subset
        cache = cache / f'{split}.pt'
        cache.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(f'{cache.resolve()}.lock'):
            if cache.exists():
                logger.info(f'loading from {cache}')
                obj = torch.load(cache, map_location=torch.device('cpu'))
            else:
                obj = []

                data = load_dataset(path=cls.name, name=cls.subset, split=split, **kwargs)
                for example in tqdm(data, desc=f'{plm.pretrained_model_name}-ing on {split} split'):
                    src = example[cls.src_key]
                    tgt = example[cls.tgt_key]

                    src, _ = plm.tokenize_as_tokens(src, max_length=cls.src_max_length)
                    tgt, _ = plm.tokenize_as_tokens(tgt, max_length=cls.tgt_max_length)
                    obj.append((src, tgt))

                logger.info(f'saving to {cache}')
                torch.save(obj, f=cache)

        if max_size is None:
            return obj
        return obj[:max_size]

    def size_of_item(self, item: Any) -> int:
        return 1

    def size_of_index(self, index: int) -> int:
        return 1

    @classmethod
    def new(cls, batch_size: int = 16, plm: Type[BartLarge] = BartLarge, **kwargs) -> List['DataLoader']:
        plm = plm(lang=cls.lang)

        pipes = [
            dict(src=TokenPipe()),
            dict(tgt=TokenPipe()),
        ]

        for ps in pipes:
            for name, pipe in ps.items():
                logger.info(f'{name} => {pipe}')

        train = cls(split='train', plm=plm, pipes=pipes, max_size=cls.train_max_size, **kwargs)
        dev = cls(split='validation', plm=plm, pipes=pipes, max_size=cls.dev_max_size, **kwargs)
        test = cls(split='test', plm=plm, pipes=pipes, max_size=cls.test_max_size, **kwargs)

        logger.info(f'len(train) => {len(train)}')
        logger.info(f'len(dev) => {len(dev)}')
        logger.info(f'len(test) => {len(test)}')

        return DataLoader.new(
            (train, dev, test), batch_size=batch_size,
            shuffle=True, section_size=1 << 31,
        )


class CNN1(AbstractSummarization):
    name = 'cnn_dailymail'
    subset = '1.0.0'
    lang = 'en'

    src_key = 'article'
    tgt_key = 'highlights'


class CNN2(AbstractSummarization):
    name = 'cnn_dailymail'
    subset = '2.0.0'
    lang = 'en'

    src_key = 'article'
    tgt_key = 'highlights'


class CNN3(AbstractSummarization):
    name = 'cnn_dailymail'
    subset = '3.0.0'
    lang = 'en'

    src_key = 'article'
    tgt_key = 'highlights'


class XSum(AbstractSummarization):
    name = 'xsum'
    lang = 'en'

    src_key = 'document'
    tgt_key = 'summary'

    dev_max_size = 1600
