from abc import ABCMeta
from logging import getLogger
from typing import Any, Dict, List, Type

from datasets import DatasetDict, load_dataset, load_from_disk

from torchglyph import DEBUG, data_dir
from torchglyph.data.abc import DataLoader, DataStore
from torchglyph.dist import get_device
from torchglyph.io import cache_folder, is_dataset_dict_folder, lock_folder
from torchglyph.nn.plm.abc import PLM
from torchglyph.nn.plm.roberta import RobertaBase

logger = getLogger(__name__)


class Glue(DataStore, metaclass=ABCMeta):
    name = 'glue'
    subset: str
    lang: str = 'en'

    train_split: str = 'train'
    validation_split: str = 'validation'
    test_split: str = 'test'

    key1: str
    key2: str
    label: str

    @classmethod
    def get_tokenize_fn(cls, plm: PLM):
        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            out = {'sentences1': plm.tokenize_batch(examples[cls.key1])}
            if cls.key2 is not None:
                out['sentences2'] = plm.tokenize_batch(examples[cls.key2])

            out['target'] = examples[cls.label]

            return out

        remove_columns = [cls.key1, cls.label] if cls.key2 is None else [cls.key1, cls.key2, cls.label]
        torch_columns = ['sentences1', 'sentences2']
        return tokenize, remove_columns, torch_columns

    @classmethod
    def predicate(cls, example) -> bool:
        return True

    @classmethod
    def get_collate_fn(cls, **kwargs):
        pass

    @classmethod
    def load(cls, plm: PLM):
        cache = cache_folder(path=data_dir / cls.name / cls.subset, tokenizer=plm.tokenizer)

        with lock_folder(path=cache):
            if not is_dataset_dict_folder(path=cache):
                if not DEBUG:
                    ds = DatasetDict(
                        train=load_dataset(cls.name, cls.subset, split=cls.train_split),
                        validation=load_dataset(cls.name, cls.subset, split=cls.validation_split),
                        test=load_dataset(cls.name, cls.subset, split=cls.test_split),
                    )
                else:
                    ds = DatasetDict(
                        train=load_dataset(cls.name, cls.subset, split=f'{cls.train_split}[:1024]'),
                        validation=load_dataset(cls.name, cls.subset, split=f'{cls.validation_split}[:1024]'),
                        test=load_dataset(cls.name, cls.subset, split=f'{cls.test_split}[:1024]'),
                    )

                tokenize_fn, remove_columns, torch_columns = cls.get_tokenize_fn(plm=plm)
                ds: DatasetDict = ds.map(tokenize_fn, batched=True, remove_columns=remove_columns)

                ds['train'] = ds['train'].filter(cls.predicate)
                ds['validation'] = ds['validation'].filter(cls.predicate)

                ds.set_format('torch', columns=torch_columns)

                logger.info(f'saving to {cache}')
                ds.save_to_disk(cache)
                return ds

        logger.info(f'loading from {cache}')
        return load_from_disk(str(cache), keep_in_memory=True)

    @classmethod
    def new(cls, batch_size: int = 1024, plm: Type[RobertaBase] = RobertaBase, **kwargs):
        plm = plm(lang=cls.lang)

        ds = cls.load(plm=plm)

        train, dev, test = DataLoader.new(
            (ds['train'], ds['validation'], ds['test']), batch_size=batch_size,
            collate_fn=cls.get_collate_fn(device=get_device()),
            drop_last=False, section_size=4096,
        )

        return (train, dev, test), plm


class MNLIm(Glue):
    subset = 'mnli'

    validation_split = 'validation_matched'
    test_split = 'test_matched'

    key1 = 'premise'
    key2 = 'hypothesis'
    label = 'label'
