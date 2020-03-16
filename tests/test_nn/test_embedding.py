from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_padded_sequence

from tests.test_nn.corpora import SeqCorpus, SubCorpus, SENTENCES
from tests.utilities import assert_pack_allclose
from torchglyph.nn import TokEmbedding, SubLstmEmbedding


@given(
    batch_first=st.booleans(),
    batch_size=st.integers(1, 12),
    char_dim=st.integers(1, 12),
    sentences=SENTENCES,
)
def test_tok_embedding(batch_first, batch_size, char_dim, sentences):
    loader, = SeqCorpus.new(sentences=sentences, batch_size=batch_size, batch_first=batch_first)
    layer = TokEmbedding(
        num_embeddings=len(loader.vocabs.pad), embedding_dim=char_dim,
    )

    for batch in loader:
        pad_encoding = layer(batch.pad)
        pad_encoding = pack_padded_sequence(
            pad_encoding, batch.seq_length,
            batch_first=batch_first, enforce_sorted=False,
        )

        pack_encoding = layer(batch.pack)

        assert_pack_allclose(pad_encoding, pack_encoding)


@given(
    batch_first=st.booleans(),
    batch_size=st.integers(1, 12),
    char_dim=st.integers(1, 12),
    hidden_dim=st.integers(1, 12),
    sentences=SENTENCES,
)
def test_sub_lstm_embedding(batch_first, batch_size, char_dim, hidden_dim, sentences):
    loader, = SubCorpus.new(sentences=sentences, batch_size=batch_size, batch_first=batch_first)
    layer = SubLstmEmbedding(
        num_embeddings=len(loader.vocabs.pad), embedding_dim=char_dim,
        hidden_dim=hidden_dim, dropout=0,
    )

    for batch in loader:
        pad_encoding = layer(batch.pad, batch.tok_length)
        pad_encoding = pack_padded_sequence(
            pad_encoding, batch.seq_length,
            batch_first=batch_first, enforce_sorted=False,
        )

        pack_encoding = layer(batch.pack, batch.tok_indices)

        assert_pack_allclose(pad_encoding, pack_encoding)