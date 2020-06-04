import logging
from pathlib import Path
from typing import Iterable, Any
from typing import Optional, List, Tuple, NamedTuple
from typing import TextIO

from tqdm import tqdm

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.formats import conllx
from torchglyph.pipe import PackedTokSeqPipe, SeqLengthTensorPipe, RawPipe, PackedTokPtrSeqPipe, PackedPtrSeqPipe, \
    ToSubList, UpdateCounter, Lift
from torchglyph.pipe import PaddedTokSeqPipe, PackedTokBlockPipe
from torchglyph.proc import ReplaceDigits, Identity, LoadGlove, LoadFastText, Prepend

logger = logging.getLogger(__name__)


class CoNLL2000Chunking(Dataset):
    urls = [
        ('https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz', 'train.txt.gz', 'train.txt'),
        ('https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz', 'test.txt.gz', 'test.txt'),
    ]

    @classmethod
    def load(cls, path: Path) -> Iterable[List[Any]]:
        for sent in tqdm(conllx.load(path, sep=' '), desc=f'reading {path}'):
            word, pos, chunk = map(list, zip(*sent))
            yield [word, pos, chunk]

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[List[int]], *args, **kwargs) -> None:
        chunk_vocab = self.pipes['chunk'].vocab.stoi
        for raw_word, raw_pos, raw_chunk, pred in \
                zip(batch.raw_word, batch.raw_pos, batch.raw_chunk, prediction):
            pred_chunk = [chunk_vocab[p] for p in pred]
            conllx.dump(zip(raw_word, raw_pos, raw_chunk, pred_chunk), fp, sep=' ')

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int],
            remove_missing: bool, device: int = -1) -> Tuple[DataLoader, ...]:
        if word_dim is not None:
            vectors = LoadGlove(name='6B', dim=word_dim, remove_missing=remove_missing)
        else:
            vectors = Identity()
        word = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(
            pre=ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + vectors,
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
            logger.info(f'{name} => {pipe}')

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
            word, pos, chunk, ner = map(list, zip(*sent))
            yield [word, pos, chunk, ner]

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[List[int]], *args, **kwargs) -> None:
        ner_vocab = self.pipes['ner'].vocab.stoi
        for raw_word, raw_pos, raw_chunk, raw_ner, pred in \
                zip(batch.raw_word, batch.raw_pos, batch.raw_chunk, batch.raw_ner, prediction):
            pred_ner = [ner_vocab[p] for p in pred]
            conllx.dump(zip(raw_word, raw_pos, raw_chunk, raw_ner, pred_ner), fp, sep=' ')

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int],
            remove_missing: bool, device: int = -1) -> Tuple[DataLoader, ...]:
        if word_dim is not None:
            vectors = LoadGlove(name='6B', dim=word_dim, remove_missing=remove_missing)
        else:
            vectors = Identity()
        word = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(
            pre=ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + vectors,
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
            logger.info(f'{name} => {pipe}')

        word.build_vocab(train, dev, test, name='word')
        char.build_vocab(train, dev, test, name='char')
        pos.build_vocab(train, name='pos')
        chunk.build_vocab(train, name='chunk')
        ner.build_vocab(train, name='ner')

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )


class SemEval2010T1NER(Dataset):
    lang: str

    @classmethod
    def load(cls, path: Path, **kwargs) -> Iterable[Any]:
        for sent in tqdm(conllx.load(path, sep='\t'), desc=f'reading {path}', unit=' sentences'):
            _, word, _, pos, _, _, head, drel, _, _, ner = map(list, zip(*sent))
            yield [word, pos, [int(h) for h in head], drel, ner]

    def dump(self, fp: TextIO, batch: NamedTuple, prediction: List[Any], *args, **kwargs) -> None:
        ner_vocab = self.pipes['ner'].vocab.stoi
        for raw_word, raw_pos, raw_ner, pred in \
                zip(batch.raw_word, batch.raw_pos, batch.raw_ner, prediction):
            assert len(raw_word) == len(raw_pos) == len(raw_ner) == len(pred)

            pred_ner = [ner_vocab[p] for p in pred]
            conllx.dump(zip(raw_word, raw_pos, raw_ner, pred_ner), fp, sep=' ')

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int],
            remove_missing: bool, device: int = -1) -> Tuple['DataLoader', ...]:
        if word_dim is not None:
            vectors = LoadFastText(str.lower, lang=cls.lang, remove_missing=remove_missing)
        else:
            vectors = Identity()
        word = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(
            pre=Prepend('<root>', 1) + ReplaceDigits(repl_token='<digits>') + ...,
            vocab=... + vectors,
        )
        length = SeqLengthTensorPipe(device=device).with_(pre=Prepend('<root>', 1) + ...)
        char = PackedTokBlockPipe(device=device, unk_token='<unk>').with_(
            pre=ToSubList() + Lift(Prepend('<root>', 1)) + Lift(UpdateCounter()),
        )
        word_ptr = PackedTokPtrSeqPipe(device=device, reverse=False).with_(pre=Prepend(0, 1) + ...)
        pos = PackedTokSeqPipe(device=device, unk_token='<unk>').with_(pre=Prepend('<root>', 1) + ...)
        head = PackedPtrSeqPipe(device=device).with_(pre=Prepend(0, 1) + ...)
        drel = PackedTokSeqPipe(device=device, unk_token='root').with_(pre=Prepend('<root>', 1) + ...)
        ner = PaddedTokSeqPipe(device=device, unk_token='O', pad_token='O')

        pipes = [
            dict(word=word, length=length, char=char, word_ptr=word_ptr, raw_word=RawPipe()),
            dict(pos=pos, raw_pos=RawPipe()),
            dict(head=head),
            dict(drel=drel, raw_drel=RawPipe()),
            dict(ner=ner, raw_ner=RawPipe()),
        ]

        train, dev, test = cls.paths()
        train = cls(path=train, pipes=pipes)
        dev = cls(path=dev, pipes=pipes)
        test = cls(path=test, pipes=pipes)

        for name, pipe in train.pipes.items():
            logger.info(f'{name} => {pipe}')

        word.build_vocab(train, dev, test, name='word')
        char.build_vocab(train, dev, test, name='char')
        pos.build_vocab(train, name='pos')
        drel.build_vocab(train, name='drel')
        ner.build_vocab(train, name='ner')

        return DataLoader.new(
            (train, dev, test),
            batch_size=batch_size, shuffle=True,
        )


class SemEval2010T1NERCatalan(SemEval2010T1NER):
    urls = [
        ('https://www.dropbox.com/s/nqedh3zmk5k80n7/train.sd.conllx?dl=1', 'train.sd.conllx'),
        ('https://www.dropbox.com/s/027umbuks3njwry/dev.sd.conllx?dl=1', 'dev.sd.conllx'),
        ('https://www.dropbox.com/s/ldwn6z1xl5vki4y/test.sd.conllx?dl=1', 'test.sd.conllx'),
    ]
    lang = 'ca'


class SemEval2010T1NERSpanish(SemEval2010T1NER):
    urls = [
        ('https://www.dropbox.com/s/lyxgvc161ai20v0/train.sd.conllx?dl=1', 'train.sd.conllx'),
        ('https://www.dropbox.com/s/8tmbi7ki6ctasez/dev.sd.conllx?dl=1', 'dev.sd.conllx'),
        ('https://www.dropbox.com/s/nnj94hdmlq3jjm8/test.sd.conllx?dl=1', 'test.sd.conllx'),
    ]
    lang = 'es'
