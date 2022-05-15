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
    'PositionalEmbedding',
    'TriangularEmbedding',
]


class TokenEmbedding(nn.Embedding, metaclass=RuaMeta):
    def __init__(self, embedding_dim: int, freeze: bool = False, *, vocab: Vocab = None,
                 num_embeddings: int = 0, padding_idx: int = None,
                 dtype: torch.dtype = torch.float32) -> None:
        if vocab is not None:
            super(TokenEmbedding, self).__init__(
                embedding_dim=embedding_dim,
                num_embeddings=len(vocab),
                padding_idx=vocab.pad_idx,
                _weight=vocab.vectors, dtype=dtype,
            )
        else:
            super(TokenEmbedding, self).__init__(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=None, dtype=dtype,
            )
        self.weight.requires_grad = not freeze

    def extra_repr(self) -> str:
        args = [super(TokenEmbedding, self).extra_repr()]
        if not self.weight.requires_grad:
            args.append('frozen')
        return ', '.join(args)

    @torch.no_grad()
    def padding_mask(self, indices: Tensor) -> Tensor:
        return indices == (self.padding_idx if self.padding_idx is None else -100)


class CharLstmEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = 50, num_layers: int = 1,
                 bias: bool = True, bidirectional: bool = True, dropout: float = 0.5, *,
                 char_embedding: TokenEmbedding, dtype: torch.dtype = torch.float32) -> None:
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
            dtype=dtype,
        )

        self.num_directions = 2 if self.rnn.bidirectional else 1
        self.embedding_dim = self.rnn.hidden_size * self.num_directions

    def forward(self, indices: PackedSequence) -> Tensor:
        embedding = self.embedding(indices)
        _, (encoding, _) = self.rnn(embedding)
        return rearrange(encoding, '(l d) b x -> l b (d x)', d=self.num_directions)[-1]


class PositionalEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 freeze: bool = False, *, dtype: torch.dtype = torch.float32) -> None:
        super(PositionalEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty((self.num_embeddings, self.embedding_dim), dtype=dtype),
            requires_grad=not freeze,
        )

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> Tensor:
        return bert_normal_(self.weight)

    def extra_repr(self) -> str:
        args = [
            f'{self.num_embeddings}',
            f'{self.embedding_dim}',
        ]
        if not self.weight.requires_grad:
            args.append('frozen')
        return ', '.join(args)

    def forward(self, sequence: Union[CattedSequence, PackedSequence]) -> Union[CattedSequence, PackedSequence]:
        if isinstance(sequence, CattedSequence):
            token_sizes = sequence.token_sizes.to(device=sequence.data.device)
            _, sequence = major_sizes_to_ptr(sizes=token_sizes)
        elif isinstance(sequence, PackedSequence):
            batch_sizes = sequence.batch_sizes.to(device=sequence.data.device)
            sequence, _ = major_sizes_to_ptr(sizes=batch_sizes)
        else:
            TypeError(f'{type(sequence)} is not supported')

        return sequence._replace(data=torch.embedding(weight=self.weight, indices=sequence))


class TriangularEmbedding(PositionalEmbedding):
    @torch.no_grad()
    def reset_parameters(self) -> None:
        tok = torch.arange(self.num_embeddings, dtype=self.weight.dtype)
        vec = torch.arange(self.embedding_dim >> 1, dtype=self.weight.dtype) << 1

        position = tok[:, None] / (10000. ** (vec[None, :] / self.embedding_dim))
        embedding = torch.stack([torch.sin(position), torch.cos(position)], dim=-1)
        self.weight.data = torch.flatten(embedding, start_dim=-2)
