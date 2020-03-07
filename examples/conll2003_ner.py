import logging
from pathlib import Path
from typing import List, Tuple, Iterable, Any

from tqdm import tqdm

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.io import conllx_iter
from torchglyph.pipe import PackedSeqPipe, PackedSubPipe, PackedSeqIndicesPipe, PaddedSeqPipe, SeqLengthPipe, RawStrPipe
from torchglyph.proc import LoadGlove, ReplaceDigits, ToLower


class CoNLL2003(Dataset):
    urls = [
        ('https://github.com/glample/tagger/raw/master/dataset/eng.train', 'train.eng'),
        ('https://github.com/glample/tagger/raw/master/dataset/eng.testa', 'dev.eng'),
        ('https://github.com/glample/tagger/raw/master/dataset/eng.testb', 'test.eng'),
    ]

    @classmethod
    def iter(cls, path: Path) -> Iterable[List[Any]]:
        for sentence in tqdm(conllx_iter(path, sep=' '), desc=f'reading {path}', unit=' sent'):
            word, pos, chunk, ner = list(zip(*sentence))
            yield [word, pos, chunk, ner]

    @classmethod
    def new(cls, batch_size: int, word_dim: int, device: int = -1) -> Tuple[DataLoader, ...]:
        word = PackedSeqPipe(device=device).with_(
            pre=ToLower() + ReplaceDigits('<digits>') + ...,
            vocab=... + LoadGlove(name='6B', dim=word_dim),
        )
        wsln = SeqLengthPipe(device=device)
        char = PackedSubPipe(device=device)
        widx = PackedSeqIndicesPipe(device=device)
        pos = PackedSeqPipe(device=device)
        chunk = PackedSeqPipe(device=device)
        ner = PaddedSeqPipe(pad_token='<pad>', device=device)

        pipes = [
            dict(word=word, wsln=wsln, char=char, widx=widx, raw_word=RawStrPipe()),
            dict(pos=pos, raw_pos=RawStrPipe()),
            dict(chunk=chunk, raw_chunk=RawStrPipe()),
            dict(ner=ner, raw_ner=RawStrPipe()),
        ]

        logging.info(f'word => {word}')
        logging.info(f'wsln => {wsln}')
        logging.info(f'char => {char}')
        logging.info(f'widx => {widx}')
        logging.info(f'pos => {pos}')
        logging.info(f'chunk => {chunk}')
        logging.info(f'ner => {ner}')

        train, dev, test = cls.paths()
        train = cls(path=train, pipes=pipes)
        dev = cls(path=dev, pipes=pipes)
        test = cls(path=test, pipes=pipes)

        word.build_vocab(train, dev, test, name='word')
        char.build_vocab(train, dev, test, name='char')
        pos.build_vocab(train, name='pos')
        chunk.build_vocab(train, name='chunk')
        ner.build_vocab(train, name='ner')

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    train, dev, test = CoNLL2003.new(batch_size=10, word_dim=50)
    logging.info(f'len(train.dataset) => {len(train.dataset)}')
    logging.info(f'len(dev.dataset) => {len(dev.dataset)}')
    logging.info(f'len(test.dataset) => {len(test.dataset)}')

    for batch in train:
        logging.info(f'batch.word => {batch.word}')
        logging.info(f'batch.wsln => {batch.wsln}')
        logging.info(f'batch.char => {batch.char}')
        logging.info(f'batch.widx => {batch.widx}')
        logging.info(f'batch.pos => {batch.pos}')
        logging.info(f'batch.chunk => {batch.chunk}')
        logging.info(f'batch.ner => {batch.ner}')
        break
