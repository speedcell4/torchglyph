import logging
from pathlib import Path
from typing import Iterable, List, Any, Tuple, Optional, NamedTuple, TextIO

from tqdm import tqdm

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.formats import conllx
from torchglyph.pipe import PackedTokSeqPipe, SeqLengthTensorPipe, RawPipe, PackedTokPtrSeqPipe
from torchglyph.pipe import PaddedTokSeqPipe, PackedTokBlockPipe
from torchglyph.proc import ToLower, ReplaceDigits, Identity, LoadGlove


class CoNLL2000Chunking(Dataset):
    urls = [
        ('https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz', 'train.txt.gz', 'train.txt'),
        ('https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz', 'test.txt.gz', 'test.txt'),
    ]

    @classmethod
    def load(cls, path: Path) -> Iterable[List[Any]]:
        for sent in tqdm(conllx.load(path, sep=' '), desc=f'reading {path}'):
            word, pos, chunk = list(zip(*sent))
            yield [word, pos, chunk]

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[List[int]], *args, **kwargs) -> None:
        chunk_vocab = self.pipes['chunk'].vocab.stoi
        for raw_word, raw_pos, raw_chunk, pred in \
                zip(batch.raw_word, batch.raw_pos, batch.raw_chunk, prediction):
            pred_chunk = [chunk_vocab[p] for p in pred]
            conllx.dump(zip(raw_word, raw_pos, raw_chunk, pred_chunk), fp, sep=' ')

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int], device: int = -1) -> Tuple[DataLoader, ...]:
        word = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(
            pre=ToLower() + ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + (Identity() if word_dim is None else LoadGlove('6B', word_dim, str.lower)),
        )
        length = SeqLengthTensorPipe(device=device)
        char = PackedTokBlockPipe(device=device, unk_token='<unk>')
        word_ptr = PackedTokPtrSeqPipe(device=device, reverse=False)
        pos = PackedTokSeqPipe(device=device, unk_token='<unk>')
        chunk = PaddedTokSeqPipe(device=device, unk_token='O', pad_token='O')

        pipes = [
            dict(word=word, length=length, char=char, word_ptr=word_ptr, raw_word=RawPipe()),
            dict(pos=pos, raw_pos=RawPipe()),
            dict(chunk=chunk, raw_chunk=RawPipe()),
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
    def load(cls, path: Path) -> Iterable[List[Any]]:
        for sent in tqdm(conllx.load(path, sep=' '), desc=f'reading {path}', unit=' sents'):
            word, pos, chunk, ner = list(zip(*sent))
            yield [word, pos, chunk, ner]

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[List[int]], *args, **kwargs) -> None:
        ner_vocab = self.pipes['ner'].vocab.stoi
        for raw_word, raw_pos, raw_chunk, raw_ner, pred in \
                zip(batch.raw_word, batch.raw_pos, batch.raw_chunk, batch.raw_ner, prediction):
            pred_ner = [ner_vocab[p] for p in pred]
            conllx.dump(zip(raw_word, raw_pos, raw_chunk, raw_ner, pred_ner), fp, sep=' ')

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int], device: int = -1) -> Tuple[DataLoader, ...]:
        word = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(
            pre=ToLower() + ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + (Identity() if word_dim is None else LoadGlove(name='6B', dim=word_dim)),
        )
        length = SeqLengthTensorPipe(device=device)
        char = PackedTokBlockPipe(device=device, unk_token='<unk>')
        word_ptr = PackedTokPtrSeqPipe(device=device, reverse=False)
        pos = PackedTokSeqPipe(device=device, unk_token='<unk>')
        chunk = PackedTokSeqPipe(device=device, unk_token='<unk>')
        ner = PaddedTokSeqPipe(device=device, unk_token='O', pad_token='O')

        pipes = [
            dict(word=word, length=length, char=char, word_ptr=word_ptr, raw_word=RawPipe()),
            dict(pos=pos, raw_pos=RawPipe()),
            dict(chunk=chunk, raw_chunk=RawPipe()),
            dict(ner=ner, raw_ner=RawPipe()),
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
