import logging
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, NewType, Tuple, Type, Union

from datasets import Dataset, DatasetDict, load_from_disk
from seqeval.metrics.sequence_labeling import get_entities
from tokenizers import Tokenizer
from torch.types import Device
from torchrua import cat_sequence

from torchglyph import data_dir
from torchglyph.data.abc import DataLoader, DataStore
from torchglyph.dist import get_device
from torchglyph.formats.conll import iter_sentence
from torchglyph.io import cache_folder, is_dataset_dict_folder, lock_folder
from torchglyph.nn.plm.abc import PLM
from torchglyph.nn.plm.roberta import RobertaBase
from torchglyph.tokenize_utils import encode_batch, get_iterator, train_word_tokenizer


def convert_scheme(tags: List[str]) -> List[str]:
    out = ['O' for _ in tags]

    for name, x, y in get_entities(tags):
        if x == y:
            out[x] = f'S-{name}'
        else:
            out[x] = f'B-{name}'
            out[y] = f'E-{name}'
            for index in range(x + 1, y):
                out[index] = f'I-{name}'

    return out


class TokenClassification(DataStore):
    lang: str

    @classmethod
    def get_tokenize_fn(cls, plm: PLM, target_tokenizer: Tokenizer, **kwargs):
        def tokenize(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
            token, segment_size = plm.tokenize_batch(examples['token'], add_prefix_space=True)

            return {
                'token': token,
                'segment_size': segment_size,
                'target': encode_batch(examples['target'], tokenizer=target_tokenizer),
                'size': [len(example) for example in examples['target']],
            }

        return tokenize

    @classmethod
    def predicate(cls, example) -> bool:
        return True

    @classmethod
    def get_collate_fn(cls, device: Device, **kwargs):
        def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, Any]:
            return {
                'token': cat_sequence([example['token'] for example in examples]).to(device=device),
                'segment_size': cat_sequence([example['segment_size'] for example in examples]).to(device=device),
                'target': cat_sequence([example['target'] for example in examples]).to(device=device),
            }

        return collate_fn

    @classmethod
    def load_split(cls, path: Path, **kwargs):
        with path.open(mode='r', encoding='utf-8') as fp:
            for token, target in iter_sentence(fp=fp, config=cls.Config, sep=' '):
                yield dict(token=token, target=convert_scheme(list(target)))

    @classmethod
    def load(cls, plm: PLM, **kwargs):
        cache = cache_folder(path=data_dir / cls.name, plm=plm.pretrained_model_name)
        tokenizer_cache = str(cache / 'tokenizer.json')

        with lock_folder(path=cache):
            if not is_dataset_dict_folder(path=cache):
                train, dev, test = cls.paths()
                ds = DatasetDict(
                    train=Dataset.from_list(list(cls.load_split(path=train))),
                    validation=Dataset.from_list(list(cls.load_split(path=dev))),
                    test=Dataset.from_list(list(cls.load_split(path=test))),
                )

                target_tokenizer = train_word_tokenizer(
                    get_iterator(ds['train'], ds['validation'], column_names=['target']),
                    pre_tokenizer=False, unk_token=None,
                )

                tokenize_fn = cls.get_tokenize_fn(plm=plm, target_tokenizer=target_tokenizer)

                ds = ds.map(tokenize_fn, batched=True)
                ds.set_format('torch', columns=['token', 'segment_size', 'target'])

                ds.save_to_disk(cache)
                target_tokenizer.save(tokenizer_cache)
                return ds, target_tokenizer

        return load_from_disk(cache, keep_in_memory=True), Tokenizer.from_file(tokenizer_cache)

    @classmethod
    def new(cls, batch_size: int = 1024, plm: Type[RobertaBase] = RobertaBase, **kwargs):
        plm = plm(lang=cls.lang)

        ds, target_tokenizer = cls.load(plm=plm, **kwargs)

        train, dev, test = DataLoader.new(
            (ds['train'], ds['validation'], ds['test']), batch_size=batch_size,
            collate_fn=cls.get_collate_fn(device=get_device()),
            drop_last=False, section_size=4096,
        )

        return (train, dev, test), (plm, target_tokenizer)


class CoNLL2003(TokenClassification):
    name = 'conll2003'
    lang = 'en'

    class Config(NamedTuple):
        token: str
        pos_: str
        chunk_: str
        target: str

    @classmethod
    def urls(cls, **kwargs) -> List[Tuple[str, ...]]:
        return [
            ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.train',),
            ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testa',),
            ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb',),
        ]


conll2003 = NewType('conll2003', CoNLL2003.new)

Data = Union[
    Type[conll2003],
]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    _ = CoNLL2003.new()
