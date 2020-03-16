import logging
from pathlib import Path
from typing import Iterable, Optional
from typing import List, Tuple, NamedTuple
from typing import TextIO

from tqdm import tqdm

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.formats import csv
from torchglyph.io import open_io
from torchglyph.pipe import PackedTokSeqPipe, TokTensorPipe, RawPipe
from torchglyph.proc import Identity, LoadGlove


class AgNews(Dataset):
    urls = [
        ('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
         'train.csv'),
        ('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv',
         'test.csv'),
        ('https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/classes.txt',
         'classes.csv'),
    ]

    @classmethod
    def load(cls, path: Path, target_vocab: Path, **kwargs) -> Iterable[Tuple[str, List[str], List[str]]]:
        with open_io(target_vocab, mode='r', encoding='utf-8') as fp:
            vocab = [token.strip() for token in fp]

        for raw in tqdm(csv.load(path), desc=f'reading {path.name}'):
            target, title, text = raw
            yield [vocab[int(target) - 1], title.split(), text.split()]

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[int], *args, **kwargs) -> None:
        vocab = self.vocabs.target
        for raw_title, raw_text, raw_target, pred in \
                zip(batch.raw_title, batch.raw_text, batch.raw_target, prediction):
            csv.dump((' '.join(raw_title), ' '.join(raw_text), raw_target, vocab.itos[pred]), fp)

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int], device: int = -1) -> Tuple['DataLoader', ...]:
        word = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(
            vocab=... + (Identity() if word_dim is None else LoadGlove(name='6B', dim=word_dim)),
        )
        target = TokTensorPipe(device=device, unk_token=None)

        pipes = [
            dict(target=target, raw_target=RawPipe()),
            dict(title=word, raw_title=RawPipe()),
            dict(text=word, raw_text=RawPipe()),
        ]

        train, test, target_vocab = cls.paths()
        train = cls(path=train, target_vocab=target_vocab, pipes=pipes)
        test = cls(path=test, target_vocab=target_vocab, pipes=pipes)

        for name, pipe in train.pipes.items():
            logging.info(f'{name} => {pipe}')

        word.build_vocab(train, test, name='word')
        target.build_vocab(train, test, name='target')

        return DataLoader.new(
            (train, test), shuffle=False,
            batch_size=batch_size,
        )
