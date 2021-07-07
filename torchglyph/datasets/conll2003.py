import logging
from pathlib import Path
from typing import Tuple, Iterable, NamedTuple

import torch

from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.format import load_conll
from torchglyph.pipe.abc import RawPipe
from torchglyph.pipe.packing import PackListStrPipe, PackListListStrPipe

__all__ = [
    'CoNLL2003',
]


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

    @classmethod
    def load(cls, path: Path, **kwargs) -> Iterable[NamedTuple]:
        yield from load_conll(path=path, config=cls.Config, sep=' ')

    @classmethod
    def new(cls, batch_size: int, *, device: torch.device, root: Path = data_dir, **kwargs) -> Tuple['DataLoader', ...]:
        WORD = PackListStrPipe(device=device, unk_token='<unk>')
        CHAR = PackListListStrPipe(device=device, unk_token='<unk>')
        TAG = PackListStrPipe(device=device, unk_token='O')

        pipes = [
            dict(word=WORD, char=CHAR, raw_word=RawPipe()),
            dict(tag=TAG, raw_tag=RawPipe()),
        ]

        for ps in pipes:
            for name, pipe in ps.items():
                logging.info(f'{name} => {pipe}')

        train, dev, test = cls.paths(root=root)
        train = cls(pipes=pipes, path=train)
        dev = cls(pipes=pipes, path=dev)
        test = cls(pipes=pipes, path=test)

        WORD.build_vocab(train, dev, test, name='word')
        CHAR.build_vocab(train, dev, test, name='char')
        TAG.build_vocab(train, dev, test, name='tag')

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size,
            shuffle=True, drop_last=False,
        )
