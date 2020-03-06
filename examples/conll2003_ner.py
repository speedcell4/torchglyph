from pathlib import Path
from typing import List, Tuple, Dict

from tqdm import tqdm

from torchglyph.dataset import Dataset, Pipeline, DataLoader
from torchglyph.io import conllx_iter
from torchglyph.pipelines import PackedSeqPipe, PackedSeqRangePipe, PaddedSeqPipe, SeqLengthPipe, PaddedSubPipe
from torchglyph.pipelines.sub_pipe import PackedSubPipe


class CoNLL2003(Dataset):
    def __init__(self, path: Path, pipelines: List[Dict[str, Pipeline]]) -> None:
        super(CoNLL2003, self).__init__(
            instances=[
                [datum for datum in zip(*sentence)]
                for sentence in tqdm(conllx_iter(path), desc=f'reading {path}', unit=' sentences')
            ],
            pipelines=pipelines,
        )

    @classmethod
    def loaders(cls, *paths: Path, batch_size: int) -> Tuple[DataLoader, ...]:
        WORD = PaddedSeqPipe(pad_token='<pad>', dim=50)
        WLEN = SeqLengthPipe()
        CHAR1 = PaddedSubPipe()
        CHAR2 = PackedSubPipe()
        WRNG = PackedSeqRangePipe()
        XPOS = PackedSeqPipe()
        TRGT = PackedSeqPipe()

        train, dev, test = tuple(cls(path, pipelines=[
            dict(word=WORD, wlen=WLEN, char1=CHAR1, char2=CHAR2, WRNG=WRNG),
            dict(xpos=XPOS),
            dict(),
            dict(),
            dict(target=TRGT),
        ]) for path in paths)

        WORD.build_vocab(train, dev, test)
        CHAR1.build_vocab(train, dev, test)
        CHAR2.build_vocab(train, dev, test)
        XPOS.build_vocab(train)
        TRGT.build_vocab(train)

        return DataLoader.loaders(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )


if __name__ == '__main__':
    path = Path('~/data/conll2003').expanduser().absolute()
    train, dev, test = path / 'train.stanford', path / 'dev.stanford', path / 'test.stanford'

    train, dev, test = CoNLL2003.loaders(train, dev, test, batch_size=10)

    print(train.dataset.pipelines['word'].vocab.stoi)

    for batch in train:
        print(f'batch.char1 => {batch.char1}')
        print(f'batch.char2 => {batch.char2}')
        break
