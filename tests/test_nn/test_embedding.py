from collections import Counter

import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence

from tests.test_nn.corpora import SeqCorpus, SubCorpus, SENTENCES
from tests.utilities import assert_pack_allclose
from torchglyph.nn import TokenEmbedding, CharLstmEmbedding
from torchglyph.nn.embedding import FrageEmbedding
from torchglyph.vocab import Vocab


@given(
    batch_first=st.booleans(),
    batch_size=st.integers(1, 12),
    char_dim=st.integers(1, 12),
    sentences=SENTENCES,
)
def test_tok_embedding(batch_first, batch_size, char_dim, sentences):
    loader, = SeqCorpus.new(sentences=sentences, batch_size=batch_size, batch_first=batch_first)
    layer = TokenEmbedding(
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
    layer = CharLstmEmbedding(
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


@given(
    num_batch=st.integers(1, 5),
    max_length=st.integers(1, 7),
    embedding_dim=st.integers(2, 5),
    num_partitions=st.integers(1, 3),
    sentence=SENTENCES,
    unk_token=st.sampled_from(['<unk>', None]),
    pad_token=st.sampled_from(['<pad>', None]),
)
def test_frage_embedding_shape(num_batch, max_length, embedding_dim, num_partitions, sentence, unk_token, pad_token):
    vocab = Vocab(Counter([w for s in sentence for w in s]), unk_token=unk_token, pad_token=pad_token)
    num_partitions = min(num_partitions, len(vocab.freq))
    embedding_dim = embedding_dim ** num_partitions

    layer = FrageEmbedding(embedding_dim, num_partitions, vocab)
    lengths = torch.randint(0, max_length, (num_batch,), dtype=torch.long) + 1
    x = pack_sequence([
        torch.randint(0, len(vocab), (length,), dtype=torch.long)
        for length in lengths.detach().cpu().tolist()
    ], enforce_sorted=False)
    y = layer(x)

    assert y.data.size() == (x.data.size(0), embedding_dim)
