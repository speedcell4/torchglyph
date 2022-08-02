import logging
from pathlib import Path
from typing import Iterable, NamedTuple, Any

import torch
from torch.types import Device
from torchglyph.proc.vocab import LoadFastText
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
        self.with_(
            vocab=... + LoadFastText(str.lower, name='cc', lang='en'),
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

    def size_of_item(self, item: Any) -> int:
        return len(item['word'])

    def size_of_index(self, index: int) -> int:
        return len(self.data['word'][index])

    @classmethod
    def load(cls, path: Path, **kwargs) -> Iterable[NamedTuple]:
        with tqdm(path.open(mode='r', encoding='Utf-8'), desc=f'loading {path}') as fp:
            for word, tag in iter_sentence(fp, config=cls.Config, sep=' '):
                yield word, tag

    @classmethod
    def new(cls, batch_size: int, device: Device = torch.device('cpu'), root: Path = data_dir, **kwargs):
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

        word.build_vocab(train)
        char.build_vocab(train)
        tag.build_vocab(train)

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size,
            shuffle=True,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    train, dev, test = CoNLL2003.new(batch_size=128, device=torch.device('cpu'))
    # for item in dev:
    #     print(item)
