import logging
from pathlib import Path
from typing import Iterable, Any
from typing import Tuple

import torch
from torch.types import Device

from torchglyph import data_dir
from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import PadListStrPipe


class WordPipe(PadListStrPipe):
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
        src_path = path.with_name(f'{path.name}.{src_lang}.txt')
        tgt_path = path.with_name(f'{path.name}.{tgt_lang}.txt')
        with src_path.open(mode='r', encoding=encoding) as src_fp:
            with tgt_path.open(mode='r', encoding=encoding) as tgt_fp:
                for src, tgt in zip(src_fp, tgt_fp):
                    yield src, tgt

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
        tgt.build_vocab_(train)

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size,
            shuffle=True, drop_last=False,
        )
