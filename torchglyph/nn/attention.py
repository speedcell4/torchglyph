from typing import Tuple, Optional

import torch
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn

__all__ = [
    'MultiHeadAttention',
]


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int = 8, head_dim: int = 64,
                 dropout: float = 0., bias: bool = True, *,
                 q_dim: int, k_dim: int, v_dim: int) -> None:
        super(MultiHeadAttention, self).__init__()

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout
        self.bias = bias
        self.tau = head_dim ** -0.5

        self.q = nn.Sequential(
            nn.Linear(q_dim, num_heads * head_dim, bias=bias),
            Rearrange('... q (h x) -> h q x', h=num_heads),
        )
        self.k = nn.Sequential(
            nn.Linear(k_dim, num_heads * head_dim, bias=bias),
            Rearrange('... k (h x) -> h x k', h=num_heads),
        )
        self.softmax = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Softmax(dim=-1),
        )

        self.v = nn.Sequential(
            nn.Linear(v_dim, num_heads * head_dim, bias=bias),
            Rearrange('... k (h x) -> h k x', h=num_heads),
        )
        self.o = nn.Sequential(
            Rearrange('... h q x -> q (h x)', h=num_heads),
            nn.Linear(num_heads * head_dim, q_dim, bias=bias),
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.q_dim}', f'{self.k_dim}', f'{self.v_dim}',
            f'{self.head_dim}(x{self.num_heads})',
            f'{self.dropout}', f'{self.bias}',
        ])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                mask: Optional[Tensor] = None) -> Tensor:
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        attention = q @ k * self.tau
        if mask is not None:
            attention, mask = torch.broadcast_tensors(attention, mask)
            attention.masked_fill_(mask=mask, value=-float('inf'))
        attention = self.softmax(attention)

        return self.o(attention @ v)

    def decode_tgt(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        q = self.q(q)
        k = self.k(k) if q.dim() == k.dim() else torch.cat([k, self.k(q)], dim=-1)
        v = self.v(v) if q.dim() == v.dim() else torch.cat([v, self.v(q)], dim=-2)

        attention = self.softmax(q @ k * self.tau)

        return self.o(attention @ v), k, v

    def decode_src(self, q: Tensor, k: Tensor, v: Tensor,
                   src_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        q = self.q(q)
        k = self.k(k) if q.dim() != k.dim() else k
        v = self.v(v) if q.dim() != v.dim() else v

        attention = q @ k * self.tau
        if src_mask is not None:
            attention, src_mask = torch.broadcast_tensors(attention, src_mask)
            attention.masked_fill_(mask=src_mask, value=-float('inf'))
        attention = self.softmax(attention)

        return self.o(attention @ v), k, v
