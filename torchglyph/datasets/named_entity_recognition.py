import logging
from pathlib import Path
from typing import Tuple, Iterable, NamedTuple

import torch

from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.format import load_conll
from torchglyph.pipe.packing import PackListStrPipe, PackListListStrPipe

__all__ = [
    'CoNLL2003',
]


class WordPipe(PackListStrPipe):
    pass


class CharPipe(PackListListStrPipe):
    pass


class TagPipe(PackListStrPipe):
    pass


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
    def new(cls, batch_size: int, *, device: torch.device,
            root: Path = data_dir, **kwargs) -> Tuple['DataLoader', ...]:
        word = WordPipe(device=device, unk_token='<unk>')
        char = CharPipe(device=device, unk_token='<unk>')
        tag = TagPipe(device=device, unk_token='O')

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    train, dev, test = CoNLL2003.new(
        batch_size=32, device=torch.device('cpu'),
    )
    print(f'train => {train}')
    print(f'dev => {dev}')
    print(f'test => {test}')
