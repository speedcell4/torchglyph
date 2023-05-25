from logging import getLogger
from typing import Any, Dict, List, Type

from datasets import DatasetDict, load_dataset, load_from_disk
from torch.types import Device
from torchrua import cat_sequence

from torchglyph import DEBUG, data_dir
from torchglyph.data.abc import DataLoader, DataStore
from torchglyph.dist import get_device
from torchglyph.io import cache_folder, is_dataset_dict_folder, lock_folder
from torchglyph.nn.plm import PLM, RobertaBase

logger = getLogger(__name__)


class Xsum(DataStore):
    name = 'xsum'
    lang = 'en'

    @classmethod
    def get_tokenize_fn(cls, plm: PLM, **kwargs):

        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            document = plm.tokenize_batch(examples['document'])
            summary = plm.tokenize_batch(examples['summary'])

            return {
                'document': document,
                'summary': summary,
                'document.size': [len(item) for item in document],
                'summary.size': [len(item) for item in summary],
            }

        return tokenize

    @classmethod
    def predicate(cls, example) -> bool:
        return True

    @classmethod
    def get_collate_fn(cls, device: Device, **kwargs):

        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                'document': cat_sequence([s['document'] for s in examples]).to(device=device),
                'summary': cat_sequence([s['summary'] for s in examples]).to(device=device),
            }

        return collate_fn

    @classmethod
    def load(cls, plm: PLM):
        cache = cache_folder(path=data_dir / cls.name, plm=plm.pretrained_model_name)

        with lock_folder(path=cache):
            if not is_dataset_dict_folder(path=cache):
                if not DEBUG:
                    ds: DatasetDict = load_dataset(cls.name, keep_in_memory=True)
                else:
                    ds = DatasetDict(
                        train=load_dataset(cls.name, split='train[:1024]'),
                        validation=load_dataset(cls.name, split='validation[:1024]'),
                        test=load_dataset(cls.name, split='test[:1024]'),
                    )

                tokenize_fn = cls.get_tokenize_fn(plm=plm)
                ds = ds.map(tokenize_fn, batched=True)
                ds['train'] = ds['train'].filter(cls.predicate)
                ds['validation'] = ds['validation'].filter(cls.predicate)

                ds.set_format('torch', columns=['document', 'summary'])

                logger.info(f'saving to {cache}')
                ds.save_to_disk(cache)
                return ds

        logger.info(f'loading from {cache}')
        return load_from_disk(str(cache), keep_in_memory=True)

    @classmethod
    def new(cls, batch_size: int = 1024, plm: Type[RobertaBase] = RobertaBase, **kwargs):
        plm = plm(lang=cls.lang)

        ds = cls.load(plm=plm)
        ds = ds.rename_columns({'summary.size': 'size'})

        train, dev, test = DataLoader.new(
            (ds['train'], ds['validation'], ds['test']), batch_size=batch_size,
            collate_fn=cls.get_collate_fn(device=get_device()),
            drop_last=False, section_size=4096,
        )

        return (train, dev, test), plm
