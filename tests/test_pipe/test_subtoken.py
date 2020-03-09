from typing import Union

import torch
from einops import rearrange
from hypothesis import given, strategies as st
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from tests.test_pipe.corpora import SubTokenCorpus, SENTENCES
from torchglyph.vocab import Vocab


class SubLstmEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, bidirectional: bool = True) -> None:
        super(SubLstmEmbedding, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab),
            embedding_dim=embedding_dim,
            padding_idx=vocab.stoi.get('<pad>', None),
        )
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim, num_layers=num_layers, bias=bias,
            batch_first=True, bidirectional=bidirectional,
        )

        self.embedding_dim = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def _padded_forward(self, sub: Tensor, tok_lengths: Tensor) -> Tensor:
        pack = pack_padded_sequence(
            rearrange(sub, 'a b x -> (a b) x'),
            rearrange(tok_lengths.clamp_min(1), 'a b -> (a b)'),
            batch_first=True, enforce_sorted=False,
        )
        embedding = pack._replace(data=self.embedding(pack.data))

        _, (encoding, _) = self.rnn.forward(embedding)
        return rearrange(encoding, '(l d) (a b) h -> l a b (d h)', a=sub.size(0), l=self.rnn.num_layers)[-1]

    def _packed_forward(self, sub: PackedSequence, tok_indices: PackedSequence) -> PackedSequence:
        embedding = sub._replace(data=self.embedding(sub.data))

        _, (encoding, _) = self.rnn(embedding)
        encoding = rearrange(encoding, '(l d) a h -> l a (d h)', l=self.rnn.num_layers)
        return tok_indices._replace(data=encoding[-1, tok_indices.data])

    def forward(self, sub: Union[Tensor, PackedSequence], *args) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(sub):
            return self._padded_forward(sub, *args)
        else:
            return self._packed_forward(sub, *args)


@given(
    batch_size=st.integers(1, 12),
    char_dim=st.integers(1, 12),
    hidden_dim=st.integers(1, 12),
    sentences=SENTENCES,
)
def test_sub_pad_pack(batch_size, char_dim, hidden_dim, sentences):
    loader, = SubTokenCorpus.new(sentences=sentences, batch_size=batch_size, word_dim=None)
    layer = SubLstmEmbedding(vocab=loader.vocabs.pad, embedding_dim=char_dim, hidden_dim=hidden_dim)

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
