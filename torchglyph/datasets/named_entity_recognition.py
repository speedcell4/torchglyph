import logging
from pathlib import Path
from typing import Iterable, NamedTuple, Any, List

import torch
from torch.types import Device
from tqdm import tqdm

from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.formats.conll import iter_sentence
from torchglyph.pipe.packing import PackedStrListPipe, PackedStrListListPipe

__all__ = [
    'CoNLL2003',
]


class WordPipe(PackedStrListPipe):
    def __init__(self, device: Device) -> None:
        super(WordPipe, self).__init__(
            device=device, dtype=torch.long,
            unk_token='<unk>', special_tokens=(), threshold=10,
        )


class CharPipe(PackedStrListListPipe):
    def __init__(self, device: Device) -> None:
        super(CharPipe, self).__init__(
            device=device, dtype=torch.long,
            unk_token='<unk>', special_tokens=(), threshold=10,
        )


class TagPipe(PackedStrListPipe):
    def __init__(self, device: Device) -> None:
        super(TagPipe, self).__init__(
            device=device, dtype=torch.long,
            unk_token='O', special_tokens=(), threshold=120,
        )


class CoNLL2003(Dataset):
    urls = [
        ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.train', 'train.txt'),
        ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testa', 'dev.txt'),
        ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb', 'test.txt'),
    ]

    class Config(NamedTuple):
        word: str
        pos_: str
        chunk_: str
        tag: str

    def get_size(self, item: Any) -> int:
        return item['word'].size()[0]

    @classmethod
    def load(cls, path: Path, **kwargs) -> Iterable[NamedTuple]:
        with tqdm(path.open(mode='r', encoding='Utf-8'), desc=f'loading {path}') as fp:
            for item in iter_sentence(fp, config=cls.Config, sep=' '):
                yield item

    @classmethod
    def new(cls, batch_size: int, *, device: Device,
            root: Path = data_dir, **kwargs) -> List['DataLoader']:
        word = WordPipe(device=device)
        char = CharPipe(device=device)
        tag = TagPipe(device=device)

        pipes = [
            dict(word=word, char=char),
            dict(tag=tag),
        ]

        for ps in pipes:
            for name, pipe in ps.items():
                logging.info(f'{name} => {pipe}')

        train, dev, test = cls.paths(root=root)
        train = cls(pipes=pipes, path=train)
        dev = cls(pipes=pipes, path=dev)
        test = cls(pipes=pipes, path=test)

        word.build_vocab_(train)
        char.build_vocab_(train)
        tag.build_vocab_(train)

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size,
            shuffle=True, drop_last=False,
        )
