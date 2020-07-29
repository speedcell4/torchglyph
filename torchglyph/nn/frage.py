import math

import torch
from torch import Tensor
from torch import nn
from torch.nn import init

from torchglyph.functional import SupportPackMeta
from torchglyph.vocab import Vocab


def partition_by_entropy(num_blocks: int, vocab: Vocab) -> Tensor:
    freq = torch.tensor([f for _, f in vocab.freq.most_common()], dtype=torch.float32)
    freq = freq / freq.sum(dim=-1, keepdim=True)
    entropy = (freq * freq.log()).neg().cumsum(dim=-1)
    threshold = torch.arange(num_blocks, dtype=torch.float32) * (entropy[-1] / num_blocks)

    return (entropy[None, :] >= threshold[:, None]).long().sum(dim=0) - 1


class FrageEmbedding(nn.Module, metaclass=SupportPackMeta):
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
