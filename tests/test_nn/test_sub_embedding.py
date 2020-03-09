import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_padded_sequence

from tests.test_nn.corpora import SubTokenCorpus, SENTENCES
from torchglyph.nn import SubLstmEmbedding


@given(
    batch_first=st.booleans(),
    batch_size=st.integers(1, 12),
    char_dim=st.integers(1, 12),
    hidden_dim=st.integers(1, 12),
    sentences=SENTENCES,
)
def test_sub_lstm_embedding(batch_first, batch_size, char_dim, hidden_dim, sentences):
    loader, = SubTokenCorpus.new(sentences=sentences, batch_size=batch_size, word_dim=None, batch_first=batch_first)
    layer = SubLstmEmbedding(vocab=loader.vocabs.pad, dim=char_dim, hidden_dim=hidden_dim)

    for batch in loader:
        pad_encoding = layer(batch.pad, batch.word_lengths)
        pad_encoding = pack_padded_sequence(
            pad_encoding, batch.seq_length,
            batch_first=batch_first, enforce_sorted=False,
        )

        pack_encoding = layer(batch.pack, batch.word_indices)

        assert torch.allclose(pad_encoding.data, pack_encoding.data, atol=1e-5)
        assert torch.allclose(pad_encoding.batch_sizes, pack_encoding.batch_sizes)
        assert torch.allclose(pad_encoding.sorted_indices, pack_encoding.sorted_indices)
        assert torch.allclose(pad_encoding.unsorted_indices, pack_encoding.unsorted_indices)
