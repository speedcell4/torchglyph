import logging
from pathlib import Path
from typing import List, Tuple, Iterable, Any

from tqdm import tqdm

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.io import conllx_iter
from torchglyph.pipe import PackedSeqPipe, PackedSubPipe, PackedSeqIndicesPipe, PaddedSeqPipe, SeqLengthPipe, RawStrPipe
from torchglyph.proc import LoadGlove, ReplaceDigits, ToLower, StatsVocab


class CoNLL2003(Dataset):
    @classmethod
    def iter(cls, path: Path) -> Iterable[List[Any]]:
        for sentence in tqdm(conllx_iter(path), desc=f'reading {path}', unit=' sentences'):
            word, pos, head, drel, ner = list(zip(*sentence))
            yield [word, pos, [int(h) for h in head], drel, ner]

    @classmethod
    def new(cls, *paths: Path, batch_size: int, device: int = -1) -> Tuple[DataLoader, ...]:
        word = PackedSeqPipe(device=device).with_(
            pre=ToLower() + ReplaceDigits('<digits>') + ...,
            vocab=... + LoadGlove(name='6B', dim=50) + StatsVocab(),
        )
        wsln = SeqLengthPipe(device=device)
        char = PackedSubPipe(device=device).with_(
            vocab=... + StatsVocab(),
        )
        widx = PackedSeqIndicesPipe(device=device)
        pos = PackedSeqPipe(device=device).with_(
            vocab=... + StatsVocab(),
        )
        ner = PaddedSeqPipe(pad_token='<pad>', device=device).with_(
            vocab=... + StatsVocab(),
        )

        print(f'word => {word}')
        print(f'wsln => {wsln}')
        print(f'char => {char}')
        print(f'widx => {widx}')
        print(f'pos => {pos}')
        print(f'ner => {ner}')

        train, dev, test = tuple(cls(path=path, pipes=[
            dict(word=word, wsln=wsln, char=char, widx=widx, raw_word=RawStrPipe()),
            dict(pos=pos, raw_pos=RawStrPipe()),
            dict(),
            dict(),
            dict(ner=ner, raw_ner=RawStrPipe()),
        ]) for path in paths)

        word.build_vocab(train, dev, test, name='word')
        char.build_vocab(train, dev, test, name='char')
        pos.build_vocab(train, name='pos')
        ner.build_vocab(train, name='ner')

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    root = Path('~/data/conll2003').expanduser().absolute()
    train, dev, test = root / 'train.stanford', root / 'dev.stanford', root / 'test.stanford'

    train, dev, test = CoNLL2003.new(train, dev, test, batch_size=10)
    print(f'len(train) => {len(train)}')
    print(f'len(dev) => {len(dev)}')
    print(f'len(test) => {len(test)}')

    for batch in train:
        print(f'batch.word => {batch.word}')
        print(f'batch.wsln => {batch.wsln}')
        print(f'batch.char => {batch.char}')
        print(f'batch.widx => {batch.widx}')
        print(f'batch.pos => {batch.pos}')
        print(f'batch.ner => {batch.ner}')
        break
