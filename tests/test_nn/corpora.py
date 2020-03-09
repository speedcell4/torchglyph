from string import ascii_letters
from typing import Iterable, List, Any, Optional, Tuple

from hypothesis import strategies as st

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.pipe import PaddedSubPipe, PaddedTokLengthPipe, PackedSubPipe, PackedTokIndicesPipe, SeqLengthPipe


class SubTokenCorpus(Dataset):
    @classmethod
    def load(cls, sentences) -> Iterable[List[Any]]:
        yield from sentences

    @classmethod
    def new(cls, sentences, batch_first: bool, batch_size: int,
            word_dim: Optional[int], device: int = -1) -> Tuple[DataLoader, ...]:
        pad = PaddedSubPipe(device=device, unk_token='<unk>', pad_token='<pad>', batch_first=batch_first)
        word_lengths = PaddedTokLengthPipe(device=device, batch_first=batch_first)

        pack = PackedSubPipe(device=device, unk_token='<unk>')
        word_indices = PackedTokIndicesPipe(device=device)

        seq_length = SeqLengthPipe(device=device)

        pipes = [
            dict(
                pad=pad, word_lengths=word_lengths,
                pack=pack, word_indices=word_indices,
                seq_length=seq_length,
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
