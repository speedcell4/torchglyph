from typing import Union

import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

from torchglyph.vocab import Vocab


class SubLstmEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, dim: int, hidden_dim: int, dropout: float, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True, bidirectional: bool = True) -> None:
        super(SubLstmEmbedding, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=dim,
            padding_idx=vocab.stoi.get('<pad>', None),
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim, num_layers=num_layers, bias=bias,
            batch_first=batch_first, bidirectional=bidirectional,
        )

        self.embedding_dim = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def _padded_forward(self, sub: Tensor, tok_lengths: Tensor) -> Tensor:
        pack = pack_padded_sequence(
            rearrange(sub, 'a b x -> (a b) x'),
            rearrange(tok_lengths.clamp_min(1), 'a b -> (a b)'),
            batch_first=self.rnn.batch_first, enforce_sorted=False,
        )
        embedding = pack._replace(data=self.dropout(self.embedding(pack.data)))

        _, (encoding, _) = self.rnn.forward(embedding)
        return rearrange(encoding, '(l d) (a b) h -> l a b (d h)', a=sub.size(0), l=self.rnn.num_layers)[-1]

    def _packed_forward(self, sub: PackedSequence, tok_indices: PackedSequence) -> PackedSequence:
        embedding = sub._replace(data=self.dropout(self.embedding(sub.data)))

        _, (encoding, _) = self.rnn(embedding)
        encoding = rearrange(encoding, '(l d) a h -> l a (d h)', l=self.rnn.num_layers)
        return tok_indices._replace(data=encoding[-1, tok_indices.data])

    def forward(self, sub: Union[Tensor, PackedSequence], *args) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(sub):
            return self._padded_forward(sub, *args)
        else:
            return self._packed_forward(sub, *args)
