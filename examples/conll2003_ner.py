from pathlib import Path
from typing import List, Tuple, Iterable, Any

from tqdm import tqdm

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.io import conllx_iter
from torchglyph.pipe import PackedSeqPipe, PackedSeqRangePipe, PaddedSeqPipe, SeqLengthPipe, PaddedSubPipe, RawStrPipe, \
    RawPackedTensorPipe
from torchglyph.pipe import PackedSubPipe, RawPaddedTensorPipe


class CoNLL2003(Dataset):
    @classmethod
    def instance_iter(cls, path: Path) -> Iterable[List[Any]]:
        for sentence in tqdm(conllx_iter(path), desc=f'reading {path}', unit=' sentences'):
            word, pos, head, drel, ner = list(zip(*sentence))
            yield [word, pos, [int(h) for h in head], drel, ner]

    @classmethod
    def dataloaders(cls, *paths: Path, batch_size: int, device: int = -1) -> Tuple[DataLoader, ...]:
        WORD = PaddedSeqPipe(pad_token='<pad>', dim=50, device=device)
        WLEN = SeqLengthPipe(device=device)
        CHAR1 = PaddedSubPipe(device=device)
        CHAR2 = PackedSubPipe(device=device)
        WRNG = PackedSeqRangePipe(device=device)
        XPOS = PackedSeqPipe(device=device)
        TRGT = PackedSeqPipe(device=device)

        train, dev, test = tuple(cls(path=path, pipes=[
            dict(word=WORD, wlen=WLEN, char1=CHAR1, char2=CHAR2, wrng=WRNG, raw_word=RawStrPipe()),
            dict(xpos=XPOS),
            dict(raw_head_pad=RawPaddedTensorPipe(device=device, pad_token=-1),
                 raw_head_pack=RawPackedTensorPipe(device=device)),
            dict(raw_drel=RawStrPipe()),
            dict(target=TRGT),
        ]) for path in paths)

        WORD.build_vocab(train, dev, test)
        CHAR1.build_vocab(train, dev, test)
        CHAR2.build_vocab(train, dev, test)
        XPOS.build_vocab(train)
        TRGT.build_vocab(train)

        return DataLoader.dataloaders(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )


if __name__ == '__main__':
    root = Path('~/data/conll2003').expanduser().absolute()
    train, dev, test = root / 'train.stanford', root / 'dev.stanford', root / 'test.stanford'

    train, dev, test = CoNLL2003.dataloaders(train, dev, test, batch_size=10)
    print(f'len(train) => {len(train)}')
    print(f'len(dev) => {len(dev)}')
    print(f'len(test) => {len(test)}')

    print(train.dataset.vocab("word").stoi)

    for batch in train:
        print(f'batch.char1 => {batch.char1}')
        print(f'batch.char2 => {batch.char2}')
        print(f'batch.wrng => {batch.wrng}')
        print(f'batch.wlen => {batch.wlen}')
        print(f'batch.raw_word => {batch.raw_word}')
        print(f'batch.raw_head_pad => {batch.raw_head_pad}')
        print(f'batch.raw_head_pack => {batch.raw_head_pack}')
        print(f'batch.raw_drel => {batch.raw_drel}')
        break
