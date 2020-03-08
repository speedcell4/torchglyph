import logging
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Optional

from tqdm import tqdm

from torchglyph import Dataset, DataLoader, PackedSeqPipe, ToLower, ReplaceDigits, LoadGlove, SeqLengthPipe, \
    PackedSubPipe, PackedSeqIndicesPipe, PaddedSeqMaskPipe, PaddedSeqPipe, RawStrPipe, Identity
from torchglyph.formats import conllx


class CoNLL2000Chunking(Dataset):
    urls = [
        ('https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz', 'train.txt.gz', 'train.txt'),
        ('https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz', 'test.txt.gz', 'test.txt'),
    ]

    @classmethod
    def iter(cls, path: Path) -> Iterable[List[Any]]:
        for sent in tqdm(conllx.load(path, sep=' '), desc=f'reading {path}', unit=' sents'):
            word, pos, chunk = list(zip(*sent))
            yield [word, pos, chunk]

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int], device: int = -1) -> Tuple[DataLoader, ...]:
        word = PackedSeqPipe(device=device, unk_token='<unk>').with_(
            pre=ToLower() + ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + (Identity() if word_dim is None else LoadGlove(name='6B', dim=word_dim)),
        )
        length = SeqLengthPipe(device=device)
        char = PackedSubPipe(device=device, unk_token='<unk>')
        word_indices = PackedSeqIndicesPipe(device=device)
        mask = PaddedSeqMaskPipe(device=device, filling_mask=True)
        pos = PackedSeqPipe(device=device, unk_token='<unk>')
        chunk = PaddedSeqPipe(unk_token='<O>', pad_token='O', device=device)

        pipes = [
            dict(word=word, length=length, char=char, word_indices=word_indices, mask=mask, raw_word=RawStrPipe()),
            dict(pos=pos, raw_pos=RawStrPipe()),
            dict(chunk=chunk, raw_chunk=RawStrPipe()),
        ]

        train, test = cls.paths()
        train = cls(path=train, pipes=pipes)
        test = cls(path=test, pipes=pipes)

        for name, pipe in train.pipes.items():
            logging.info(f'{name} => {pipe}')

        word.build_vocab(train, test, name='word')
        char.build_vocab(train, test, name='char')
        pos.build_vocab(train, name='pos')
        chunk.build_vocab(train, name='chunk')

        return DataLoader.new(
            (train, test),
            batch_size=batch_size, shuffle=True,
        )


class CoNLL2003NER(Dataset):
    urls = [
        ('https://github.com/glample/tagger/raw/master/dataset/eng.train', 'train.eng'),
        ('https://github.com/glample/tagger/raw/master/dataset/eng.testa', 'dev.eng'),
        ('https://github.com/glample/tagger/raw/master/dataset/eng.testb', 'test.eng'),
    ]

    @classmethod
    def iter(cls, path: Path) -> Iterable[List[Any]]:
        for sent in tqdm(conllx.load(path, sep=' '), desc=f'reading {path}', unit=' sents'):
            word, pos, chunk, ner = list(zip(*sent))
            yield [word, pos, chunk, ner]

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int], device: int = -1) -> Tuple[DataLoader, ...]:
        word = PackedSeqPipe(device=device, unk_token='<unk>').with_(
            pre=ToLower() + ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + (Identity() if word_dim is None else LoadGlove(name='6B', dim=word_dim)),
        )
        length = SeqLengthPipe(device=device)
        char = PackedSubPipe(device=device, unk_token='<unk>')
        word_indices = PackedSeqIndicesPipe(device=device)
        mask = PaddedSeqMaskPipe(device=device, filling_mask=True)
        pos = PackedSeqPipe(device=device, unk_token='<unk>')
        chunk = PackedSeqPipe(device=device, unk_token='<unk>')
        ner = PaddedSeqPipe(unk_token='O', pad_token='O', device=device)

        pipes = [
            dict(word=word, length=length, char=char, word_indices=word_indices, mask=mask, raw_word=RawStrPipe()),
            dict(pos=pos, raw_pos=RawStrPipe()),
            dict(chunk=chunk, raw_chunk=RawStrPipe()),
            dict(ner=ner, raw_ner=RawStrPipe()),
        ]

        train, dev, test = cls.paths()
        train = cls(path=train, pipes=pipes)
        dev = cls(path=dev, pipes=pipes)
        test = cls(path=test, pipes=pipes)

        for name, pipe in train.pipes.items():
            logging.info(f'{name} => {pipe}')

        word.build_vocab(train, dev, test, name='word')
        char.build_vocab(train, dev, test, name='char')
        pos.build_vocab(train, name='pos')
        chunk.build_vocab(train, name='chunk')
        ner.build_vocab(train, name='ner')

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )
