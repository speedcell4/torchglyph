from pathlib import Path
from typing import Iterable, Any
from typing import Tuple

import torch

from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import PackListStrPipe, PackListListStrPipe
from torchglyph.pipe.abc import RawPipe

__all__ = [
    'CoNLL2003',
]


class CoNLL2003(Dataset):
    url = [
        ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.train', 'train.txt'),
        ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testa', 'dev.txt'),
        ('https://raw.githubusercontent.com/glample/tagger/master/dataset/eng.testb', 'test.txt'),
    ]

    @classmethod
    def load(cls, path: Path, **kwargs) -> Iterable[Any]:
        with path.open(mode='r', encoding='utf-8') as fp:
            sentence = []
            for raw in fp:
                raw = raw.strip()
                if len(raw) != 0:
                    sentence.append(raw.split(sep=' '))
                elif len(sentence) != 0:
                    words, _, _, tags = zip(*sentence)
                    yield words, tags
                    sentence = []

            if len(sentence) != 0:
                words, _, _, tags = zip(*sentence)
                yield words, tags

    @classmethod
    def new(cls, batch_size: int, *, device: torch.device, root: Path = data_dir, **kwargs) -> Tuple['DataLoader', ...]:
        WORD = PackListStrPipe(device=device, unk_token='<unk>')
        CHAR = PackListListStrPipe(device=device, unk_token='<unk>')
        TAG = PackListStrPipe(device=device, unk_token='O')

        pipes = [
            dict(word=WORD, char=CHAR, raw_word=RawPipe()),
            dict(tag=TAG, raw_tag=RawPipe()),
        ]

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
