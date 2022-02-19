from typing import Union

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchrua import RuaMeta, RuaSequential
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
    def __init__(self, embedding_dim: int, num_embeddings: int = 1024, freeze: bool = False, *,
                 dtype: torch.dtype = torch.float32) -> None:
        super(PositionalEmbedding, self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(self.obtain_parameters(dtype=dtype), requires_grad=not freeze)

    def extra_repr(self) -> str:
        args = [
            f'{self.num_embeddings}',
            f'{self.embedding_dim}',
        ]
        if not self.weight.requires_grad:
            args.append('frozen')
        return ', '.join(args)

    @torch.no_grad()
    def obtain_parameters(self, dtype: torch.dtype, **kwargs) -> Tensor:
        return bert_normal_(torch.empty((self.num_embeddings, self.embedding_dim), dtype=dtype))

    def forward(self, indices: Union[Tensor, PackedSequence], dim: int = -2) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(indices):
            return self.weight[:indices.size()[dim]]

        batch_sizes = indices.batch_sizes.to(device=indices.data.device)
        indices, _ = major_sizes_to_ptr(batch_sizes=batch_sizes)
        return torch.embedding(weight=self.weight, indices=indices)


class TriangularEmbedding(PositionalEmbedding):
    def __init__(self, embedding_dim: int, num_embeddings: int = 1024, freeze: bool = True, *,
                 dtype: torch.dtype = torch.float32) -> None:
        super(TriangularEmbedding, self).__init__(
            embedding_dim=embedding_dim,
            num_embeddings=num_embeddings,
            freeze=freeze, dtype=dtype,
        )

    @torch.no_grad()
    def obtain_parameters(self, dtype: torch.dtype, **kwargs) -> Tensor:
        token = torch.arange(self.num_embeddings, dtype=dtype)
        feature = torch.arange(self.embedding_dim // 2, dtype=dtype) * 2

        position = token[:, None] / (10000. ** (feature[None, :] / self.embedding_dim))
        embedding = torch.stack([torch.sin(position), torch.cos(position)], dim=-1)
        return torch.flatten(embedding, start_dim=-2)
