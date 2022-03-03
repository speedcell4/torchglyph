import logging
from pathlib import Path
from typing import Iterable, Any
from typing import Tuple

import torch
from torch.types import Device
from tqdm import tqdm

from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import PaddedStrListPipe

__all__ = [
    'MachineTranslation',
    'IWSLT14',
]


class WordPipe(PaddedStrListPipe):
    def __init__(self, device: Device) -> None:
        super(WordPipe, self).__init__(
            batch_first=True, device=device,
            unk_token='<unk>', pad_token='<pad>',
            special_tokens=('<bos>', '<eos>'),
            threshold=16, dtype=torch.long,
        )


class MachineTranslation(Dataset):
    @classmethod
    def load(cls, path: Path, src_lang: str, tgt_lang: str, encoding: str = 'utf-8', **kwargs) -> Iterable[Any]:
        src_path = path.with_name(f'{path.name}.{src_lang}')
        tgt_path = path.with_name(f'{path.name}.{tgt_lang}')
        with src_path.open(mode='r', encoding=encoding) as src_fp:
            with tgt_path.open(mode='r', encoding=encoding) as tgt_fp:
                for src, tgt in tqdm(zip(src_fp, tgt_fp), desc=f'{path.resolve()}'):
                    yield [src.strip().split(' '), tgt.strip().split(' ')]

    @classmethod
    def new(cls, batch_size: int, share_vocab: bool, src_lang: str, tgt_lang: str, *,
            device: Device, root: Path = data_dir, **kwargs) -> Tuple['DataLoader', ...]:
        if share_vocab:
            src = tgt = WordPipe(device=device)
        else:
            src = WordPipe(device=device)
            tgt = WordPipe(device=device)

        pipes = [
            dict(src=src),
            dict(tgt=tgt),
        ]

        for ps in pipes:
            for name, pipe in ps.items():
                logging.info(f'{name} => {pipe}')

        train, dev, test = cls.paths(root=root)

        train = cls(pipes=pipes, path=train, src_lang=src_lang, tgt_lang=tgt_lang)
        dev = cls(pipes=pipes, path=dev, src_lang=src_lang, tgt_lang=tgt_lang)
        test = cls(pipes=pipes, path=test, src_lang=src_lang, tgt_lang=tgt_lang)

        src.build_vocab_(train)
        if not share_vocab:
            tgt.build_vocab_(train)

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size,
            shuffle=True, drop_last=False,
        )


class IWSLT14(MachineTranslation):
    urls = [(
        'https://raw.githubusercontent.com/pytorch/fairseq/master/examples/translation/prepare-iwslt14.sh',
        'prepare-iwslt14.sh',
    )]

    @classmethod
    def paths(cls, root: Path = data_dir, **kwargs) -> Tuple[Path, ...]:
        path, = super(IWSLT14, cls).paths(root=root, **kwargs)
        train = path.parent / 'iwslt14.tokenized.de-en' / 'train'
        dev = path.parent / 'iwslt14.tokenized.de-en' / 'valid'
        test = path.parent / 'iwslt14.tokenized.de-en' / 'test'
        return train, dev, test


if __name__ == '__main__':
    train, dev, test = IWSLT14.new(
        batch_size=32, share_vocab=False,
        src_lang='en', tgt_lang='de',
        device=torch.device('cpu'),
    )
    for item in train:
        print(item)
