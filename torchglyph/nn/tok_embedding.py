from typing import Union

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from torchglyph.vocab import Vocab


class TokEmbedding(nn.Module):
    def __init__(self, vocab: Vocab, dim: int, freeze: bool = False) -> None:
        super(TokEmbedding, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=len(vocab), embedding_dim=dim,
            padding_idx=vocab.stoi.get('<pad>', None),
            _weight=vocab.vectors,
        )
        if freeze:
            self.embedding.weight.requires_grad_(False)

        self.embedding_dim = dim

    def _padded_forward(self, tok: Tensor) -> Tensor:
        return self.embedding(tok)

    def _packed_forward(self, tok: PackedSequence) -> PackedSequence:
        return tok._replace(data=self.embedding(tok.data))

    def forward(self, tok: Union[Tensor, PackedSequence]) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(tok):
            return self._padded_forward(tok)
        else:
            return self._packed_forward(tok)
