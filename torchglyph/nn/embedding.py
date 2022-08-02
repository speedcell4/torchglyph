from typing import Union

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchrua import RuaMeta, RuaSequential, CattedSequence
from torchrua import major_sizes_to_ptr

from torchglyph.nn.init import bert_normal_
from torchglyph.vocab import Vocab

__all__ = [
    'TokenEmbedding',
    'CharLstmEmbedding',
    'PositionEmbedding',
    'TriangularEmbedding',
]


class TokenEmbedding(nn.Embedding, metaclass=RuaMeta):
    def __init__(self, embedding_dim: int, freeze: bool = False, *,
                 num_embeddings: int, padding_idx: int = None) -> None:
        super(TokenEmbedding, self).__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        if freeze:
            self.weight.requires_grad_(False)

    def extra_repr(self) -> str:
        args = [super(TokenEmbedding, self).extra_repr()]
        if not self.weight.requires_grad:
            args.append('frozen')
        return ', '.join(args)

    @classmethod
    def from_vocab(cls, vocab: Vocab, freeze: bool = False):
        return cls.from_pretrained(
            embeddings=vocab.weight,
            freeze=freeze, padding_idx=vocab.pad_idx,
        )


class CharLstmEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = 50, num_layers: int = 1,
                 bias: bool = True, bidirectional: bool = True, dropout: float = 0.5, *,
                 char_embedding: TokenEmbedding) -> None:
        super(CharLstmEmbedding, self).__init__()

        self.embedding = RuaSequential(
            char_embedding,
            nn.Dropout(dropout),
        )

        self.rnn = nn.LSTM(
            input_size=char_embedding.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers, bias=bias,
            bidirectional=bidirectional,
        )

        self.num_directions = 2 if self.rnn.bidirectional else 1
        self.embedding_dim = self.rnn.hidden_size * self.num_directions

    def forward(self, indices: PackedSequence) -> Tensor:
        embedding = self.embedding(indices)
        _, (encoding, _) = self.rnn(embedding)
        return rearrange(encoding, '(l d) b x -> l b (d x)', d=self.num_directions)[-1]


class PositionEmbedding(TokenEmbedding):
    def __init__(self, max_position_embeddings: int, embedding_dim: int, freeze: bool = True) -> None:
        super(PositionEmbedding, self).__init__(
            num_embeddings=max_position_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=None,
            freeze=freeze,
        )

    @torch.no_grad()
    def reset_parameters(self) -> Tensor:
        return bert_normal_(self.weight)

    def forward(self, sequence: Union[CattedSequence, PackedSequence]) -> Union[CattedSequence, PackedSequence]:
        if isinstance(sequence, CattedSequence):
            token_sizes = sequence.token_sizes.to(device=sequence.data.device)
            _, indices = major_sizes_to_ptr(sizes=token_sizes)
        elif isinstance(sequence, PackedSequence):
            batch_sizes = sequence.batch_sizes.to(device=sequence.data.device)
            indices, _ = major_sizes_to_ptr(sizes=batch_sizes)
        else:
            raise TypeError(f'{type(sequence)} is not supported')

        data = super(PositionEmbedding, self).forward(indices)
        return sequence._replace(data=data)


class TriangularEmbedding(PositionEmbedding):
    @torch.no_grad()
    def reset_parameters(self) -> None:
        index1 = torch.arange(self.num_embeddings, dtype=self.weight.dtype)
        index2 = torch.arange(self.embedding_dim >> 1, dtype=self.weight.dtype) << 1

        position = index1[:, None] / (10000. ** (index2[None, :] / self.embedding_dim))
        embedding = torch.stack([torch.sin(position), torch.cos(position)], dim=-1)
        self.weight.data = torch.flatten(embedding, start_dim=-2)
