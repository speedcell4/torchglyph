from string import ascii_letters
from typing import Iterable, List, Any, Tuple

from hypothesis import strategies as st

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import PackedTokPtrSeqPipe, SeqLengthTensorPipe, PackedTokSeqPipe, PaddedTokSeqPipe, RawPipe, \
    PaddedTokLengthPipe
from torchglyph.pipe import PaddedTokBlockPipe, PackedTokBlockPipe


class HypothesisCorpus(Dataset):
    @classmethod
    def load(cls, sentences) -> Iterable[List[Any]]:
        for sentence in sentences:
            yield [sentence]


class SeqCorpus(HypothesisCorpus):
    @classmethod
    def new(cls, sentences, batch_first: bool, batch_size: int, device: int = -1) -> Tuple[DataLoader, ...]:
        pad = PaddedTokSeqPipe(device=device, unk_token='<unk>', pad_token='<pad>', batch_first=batch_first)
        pack = PackedTokSeqPipe(device=device, unk_token='<unk>')
        seq_length = SeqLengthTensorPipe(device=device)

        pipes = [
            dict(
                pad=pad, pack=pack,
                seq_length=seq_length,
                raw=RawPipe(),
            ),
        ]

        data = cls(sentences=sentences, pipes=pipes)

        pad.build_vocab(data, name='data')
        data.pipes['pack'].vocab = data.pipes['pad'].vocab

        return DataLoader.new(
            (data,),
            batch_size=batch_size, shuffle=True,
        )


class SubCorpus(HypothesisCorpus):
    @classmethod
    def new(cls, sentences, batch_first: bool, batch_size: int, device: int = -1) -> Tuple[DataLoader, ...]:
        pad = PaddedTokBlockPipe(device=device, unk_token='<unk>', pad_token='<pad>', batch_first=batch_first)
        tok_indices = PackedTokPtrSeqPipe(device=device)
        pack = PackedTokBlockPipe(device=device, unk_token='<unk>')
        tok_length = PaddedTokLengthPipe(device=device, batch_first=batch_first)
        seq_length = SeqLengthTensorPipe(device=device)

        pipes = [
            dict(
                pad=pad, tok_length=tok_length,
                pack=pack, tok_indices=tok_indices,
                seq_length=seq_length,
                raw=RawPipe(),
            ),
        ]

        data = cls(sentences=sentences, pipes=pipes)

        pad.build_vocab(data, name='data')
        data.pipes['pack'].vocab = data.pipes['pad'].vocab

        return DataLoader.new(
            (data,),
            batch_size=batch_size, shuffle=True,
        )


TOKEN = st.text(ascii_letters, min_size=1, max_size=12)
SENTENCE = st.lists(TOKEN, min_size=1, max_size=12)
SENTENCES = st.lists(SENTENCE, min_size=1, max_size=120)
