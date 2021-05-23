import math
from typing import Union, Tuple

import torch
from einops import rearrange
from torch import Tensor
from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import PackedSequence, pack_sequence, pack_padded_sequence
from torchrua import PackedMeta

from torchglyph.nn.utils import partition_by_entropy
from torchglyph.vocab import Vocab


class TokEmbedding(nn.Embedding, metaclass=PackedMeta):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = None, unk_idx: int = None,
                 max_norm: float = None, norm_type: float = 2., scale_grad_by_freq: bool = False,
                 sparse: bool = False, _weight: Tensor = None):
        super(TokEmbedding, self).__init__(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim,
            padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight,
        )
        self.unk_idx = unk_idx

    @property
    def unk(self) -> Tensor:
        return self.weight[self.unk_idx]


class SubLstmEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 hidden_dim: int, dropout: float, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True,
                 bidirectional: bool = True, padding_idx: int = None, unk_idx: int = None) -> None:
        super(SubLstmEmbedding, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim, num_layers=num_layers, bias=bias,
            batch_first=batch_first, bidirectional=bidirectional,
        )

        self.embedding_dim = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)
        self.unk_idx = unk_idx

    @property
    def unk(self) -> Tensor:
        embedding = self.embedding.weight[None, self.unk_idx]
        _, (encoding, _) = self.rnn(pack_sequence([embedding], enforce_sorted=True))
        return rearrange(encoding, '(l d) a h -> l a (d h)', l=self.rnn.num_layers)[0, 0, :]

    def _padded_forward(self, sub: Tensor, tok_lengths: Tensor) -> Tensor:
        pack = pack_padded_sequence(
            rearrange(sub, 'a b x -> (a b) x'),
            rearrange(tok_lengths.clamp_min(1), 'a b -> (a b)'),
            batch_first=self.rnn.batch_first, enforce_sorted=False,
        )

        embedding = pack._replace(data=self.dropout(self.embedding(pack.data)))
        _, (encoding, _) = self.rnn(embedding)

        return rearrange(encoding, '(l d) (a b) h -> l a b (d h)', l=self.rnn.num_layers, a=sub.size(0))[-1]

    def _packed_forward(self, sub: PackedSequence, tok_ptr: PackedSequence) -> PackedSequence:
        embedding = sub._replace(data=self.dropout(self.embedding(sub.data)))
        _, (encoding, _) = self.rnn(embedding)

        encoding = rearrange(encoding, '(l d) a h -> l a (d h)', l=self.rnn.num_layers)
        return tok_ptr._replace(data=encoding[-1, tok_ptr.data])

    def forward(self, sub: Union[Tensor, PackedSequence], *args) -> Union[Tensor, PackedSequence]:
        if torch.is_tensor(sub):
            return self._padded_forward(sub, *args)
        else:
            return self._packed_forward(sub, *args)


class ContiguousSubLstmEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int,
                 hidden_dim: int, dropout: float, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = True,
                 bidirectional: bool = True, padding_idx: int = None) -> None:
        super(ContiguousSubLstmEmbedding, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_dim, num_layers=num_layers, bias=bias,
            batch_first=batch_first, bidirectional=bidirectional,
        )

        self.embedding_dim = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def forward(self, sub: PackedSequence, indices: Tuple[PackedSequence, PackedSequence]) -> PackedSequence:
        embedding = sub._replace(data=self.dropout(self.embedding(sub.data)))
        encoding, _ = self.rnn(embedding)  # type: (PackedSequence, _)

        fidx, bidx = indices
        fenc, benc = encoding.data.chunk(2, dim=-1)
        return fidx._replace(data=torch.cat([fenc[fidx.data], benc[bidx.data]], dim=-1))


class FrageEmbedding(nn.Module, metaclass=PackedMeta):
    def __init__(self, embedding_dim: int, num_partitions: int, vocab: Vocab) -> None:
        super(FrageEmbedding, self).__init__()

        self.num_embeddings = len(vocab)
        self.embedding_dim = embedding_dim
        self.num_partitions = num_partitions

        partitions = partition_by_entropy(num_partitions, vocab)
        partitions = torch.cat([
            torch.zeros((len(vocab) - partitions.size(0),), dtype=torch.long), partitions], dim=0)
        self.register_buffer('partitions', partitions)

        self.partition_sizes = [
            (index == self.partitions).long().sum().detach().cpu().item()
            for index in range(self.num_partitions)
        ]

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=vocab.pad_idx,
            _weight=vocab.vectors,
        )
        self.weight = nn.Parameter(
            torch.zeros((num_partitions, embedding_dim, embedding_dim), dtype=torch.float32),
            requires_grad=True,
        )

        self.register_buffer('mask', torch.zeros((num_partitions, 1, embedding_dim), dtype=torch.float32))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        embedding_dim = self.embedding_dim
        for index in range(self.num_partitions):
            init.kaiming_uniform_(self.weight[index, :, :embedding_dim], a=math.sqrt(5))
            init.constant_(self.mask[index, :, :embedding_dim], 1.)
            init.constant_(self.mask[index, :, embedding_dim:], 0.)
            embedding_dim //= 2

    def extra_repr(self) -> str:
        partition_sizes = ', '.join(map(str, self.partition_sizes))
        return f'partition_sizes=[{partition_sizes}]'

    def __repr__(self) -> str:
        args = ', '.join([self.embedding.extra_repr(), self.extra_repr()])
        return f'{self.__class__.__name__}({args})'

    def forward(self, x: Tensor) -> Tensor:
        weight = (self.weight * self.mask)[self.partitions[x]]
        return torch.einsum('...x,...zx->...z', self.embedding(x), weight)


class TokenDropout(nn.Module, metaclass=PackedMeta):
    def __init__(self, vocab: Vocab, repl_idx: int = None) -> None:
        super(TokenDropout, self).__init__()
        freq = torch.tensor([vocab.freq.get(token, 1) for token in vocab.stoi], dtype=torch.float32)
        freq = 1. / (1. + freq)
        freq[:len(vocab.special_tokens)] = 0.

        if repl_idx is None:
            assert vocab.unk_idx is not None
            repl_idx = vocab.unk_idx

        self.repl_idx = repl_idx
        self.register_buffer('freq', freq)

    def extra_repr(self) -> str:
        if self.freq.size(0) <= 10:
            return ', '.join(str(f) for f in self.freq.detach().cpu().tolist())
        else:
            freq1 = ', '.join(str(f) for f in self.freq[:+5].detach().cpu().tolist())
            freq2 = ', '.join(str(f) for f in self.freq[-5:].detach().cpu().tolist())
            return f'{freq1}, ..., {freq2}'

    def forward(self, indices: Tensor) -> Tensor:
        mask = torch.rand_like(indices, dtype=torch.float32) < self.freq[indices]
        return torch.masked_fill(indices, mask=mask, value=self.repl_idx)
