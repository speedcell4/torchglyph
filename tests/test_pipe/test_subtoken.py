from pathlib import Path
from typing import Iterable, Any
from typing import Union, Optional, List, Tuple

import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from torchglyph.dataset import Dataset, DataLoader
from torchglyph.formats import conllx
from torchglyph.pipe import PackedTokIndicesPipe, PaddedTokLengthPipe, SeqLengthPipe
from torchglyph.pipe import PaddedSubPipe, PackedSubPipe
from torchglyph.vocab import Vocab


class CoNLL2000Chunking(Dataset):
    urls = [
        ('https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz', 'test.txt.gz', 'test.txt'),
    ]

    @classmethod
    def load(cls, path: Path) -> Iterable[List[Any]]:
        cnt = 0
        for sent in conllx.load(path, sep=' '):
            word, _, _ = list(zip(*sent))
            yield [word]
            cnt += 1
            if cnt > 10:
                break

    @classmethod
    def new(cls, batch_size: int, word_dim: Optional[int], device: int = -1) -> Tuple[DataLoader, ...]:
        pad = PaddedSubPipe(device=device, unk_token='<unk>', pad_token='<pad>')
        word_lengths = PaddedTokLengthPipe(device=device, batch_first=True)

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

        data, = cls.paths()
        data = cls(path=data, pipes=pipes)

        pad.build_vocab(data, name='data')
        data.pipes['pack'].vocab = data.pipes['pad'].vocab

        return DataLoader.new(
            (data,),
            batch_size=batch_size, shuffle=True,
        )


class CharLstmEncoder(nn.Module):
    def __init__(self, vocab: Vocab, char_dim: int, hidden_dim: int):
        super(CharLstmEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=char_dim,
            padding_idx=vocab.stoi.get('<pad>', None),
        )
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim, num_layers=1, bias=True,
            batch_first=True, bidirectional=True,
        )

    def _pad_forward(self, char: Tensor, word_lengths: Tensor) -> Tensor:
        pack = pack_padded_sequence(
            rearrange(char, 'b s x -> (b s) x'),
            rearrange(word_lengths.clamp_min(1), 'b s -> (b s)'),
            batch_first=True, enforce_sorted=False,
        )
        embedding = pack._replace(data=self.embedding(pack.data))

        _, (encoding, _) = self.rnn.forward(embedding)
        return rearrange(encoding, '(l d) (b s) h -> l b s (d h)', b=char.size(0), l=self.rnn.num_layers)[-1]

    def _pack_forward(self, char: PackedSequence, word_indices: PackedSequence):
        embedding = char._replace(data=self.embedding(char.data))

        _, (encoding, _) = self.rnn(embedding)
        encoding = rearrange(encoding, '(l d) b h -> l b (d h)', l=self.rnn.num_layers)
        return word_indices._replace(data=encoding[-1, word_indices.data])

    def forward(self, char: Union[PackedSequence, Tensor], *args) -> Union[PackedSequence, Tensor]:
        if isinstance(char, Tensor):
            return self._pad_forward(char, *args)
        return self._pack_forward(char, *args)


@given(
    batch_size=st.integers(1, 12),
    char_dim=st.integers(1, 12),
    hidden_dim=st.integers(1, 12),
)
def test_sub_pad_pack(batch_size, char_dim, hidden_dim):
    loader, = CoNLL2000Chunking.new(batch_size=batch_size, word_dim=None)
    layer = CharLstmEncoder(loader.vocabs.pad, char_dim=char_dim, hidden_dim=hidden_dim)

    for batch in loader:
        pad_encoding = layer(batch.pad, batch.word_lengths)
        pad_encoding = pack_padded_sequence(
            pad_encoding, batch.seq_length,
            batch_first=True, enforce_sorted=False,
        )

        pack_encoding = layer(batch.pack, batch.word_indices)

        assert torch.allclose(pad_encoding.data, pack_encoding.data, atol=1e-5)
        assert torch.allclose(pad_encoding.batch_sizes, pack_encoding.batch_sizes)
        assert torch.allclose(pad_encoding.sorted_indices, pack_encoding.sorted_indices)
        assert torch.allclose(pad_encoding.unsorted_indices, pack_encoding.unsorted_indices)
