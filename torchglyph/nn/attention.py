from abc import ABCMeta
from typing import Tuple, Type, Union

import torch
from torch import Tensor, nn

Cache = Tuple[Tensor, Tensor]


def dot(*, q: Tensor, k: Tensor, v: Tensor, tau: float, mask: Tensor = None) -> Tensor:
    s = torch.einsum('...qhx,...khx->...hqk', q, k) * tau
    if mask is not None:
        s, mask = torch.broadcast_tensors(s, mask[..., None, :, :])
        s = torch.masked_fill(s, mask=mask, value=-float('inf'))

    a = (s - s.logsumexp(dim=-1, keepdim=True)).exp()
    return torch.einsum('...hqk,...khy->...qhy', a, v)


def detach(*, q: Tensor, k: Tensor, v: Tensor, tau: float, mask: Tensor = None) -> Tensor:
    s = torch.einsum('...qhx,...khx->...hqk', q, k) * tau
    if mask is not None:
        s, mask = torch.broadcast_tensors(s, mask[..., None, :, :])
        s = torch.masked_fill(s, mask=mask, value=-float('inf'))

    a = (s - s.logsumexp(dim=-1, keepdim=True).detach()).exp()
    return torch.einsum('...hqk,...khy->...qhy', a, v)


class MultiAttentionMeta(nn.Module, metaclass=ABCMeta):
    def __init__(self, algo: Union[Type[dot], Type[detach]] = dot,
                 num_heads: int = 8, bias: bool = True, dropout: float = 0.0, *,
                 q_dim: int, k_dim: int, v_dim: int, o_dim: int, **kwargs) -> None:
        super(MultiAttentionMeta, self).__init__()

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.o_dim = o_dim

        self.bias = bias
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = (o_dim + num_heads - 1) // num_heads
        self.tau = self.head_dim ** -0.5

        self.algo = algo

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'q={self.q_dim}', f'k={self.k_dim}', f'v={self.v_dim}',
            f'num_heads={self.num_heads}', f'head_dim={self.head_dim}',
            f'bias={self.bias}', f'dropout={self.dropout}',
        ])

    def forward_qkv(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, *args, mask: Tensor = None, **kwargs) -> Tuple[Tensor, Tensor, Cache]:
        q, k, v = self.forward_qkv(*args, **kwargs)

        z = self.algo(q=q, k=k, v=v, tau=self.tau, mask=mask)
        o = self.o(z.flatten(start_dim=-2))

        return o, z, (k, v)


class MultiAttention(MultiAttentionMeta):
    def __init__(self, algo: Union[Type[dot], Type[detach]] = dot,
                 num_heads: int = 8, bias: bool = True, dropout: float = 0.0, *,
                 q_dim: int, k_dim: int, v_dim: int, o_dim: int, **kwargs) -> None:
        super(MultiAttention, self).__init__(
            algo=algo, bias=bias, num_heads=num_heads,
            q_dim=q_dim, k_dim=k_dim, v_dim=v_dim,
            o_dim=o_dim, dropout=dropout,
        )

        self.q = nn.Linear(q_dim, num_heads * self.head_dim, bias=bias)
        self.k = nn.Linear(k_dim, num_heads * self.head_dim, bias=bias)
        self.v = nn.Linear(v_dim, num_heads * self.head_dim, bias=bias)

        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(p=dropout),
        )

        self.o = nn.Linear(num_heads * self.head_dim, o_dim, bias=bias)

    def forward_qkv(self, q: Tensor, k: Tensor, v: Tensor, cache: Cache = None, **kwargs):
        assert cache is None

        q = self.q(q).unflatten(dim=-1, sizes=(self.num_heads, self.head_dim))
        k = self.k(k).unflatten(dim=-1, sizes=(self.num_heads, self.head_dim))
        v = self.v(v).unflatten(dim=-1, sizes=(self.num_heads, self.head_dim))

        return q, k, v


class SelfAttention(MultiAttentionMeta):
    def __init__(self, algo: Union[Type[dot], Type[detach]] = dot,
                 num_heads: int = 8, bias: bool = True, dropout: float = 0.0, *,
                 q_dim: int, **kwargs) -> None:
        super(SelfAttention, self).__init__(
            algo=algo, bias=bias, num_heads=num_heads,
            q_dim=q_dim, k_dim=q_dim, v_dim=q_dim,
            o_dim=q_dim, dropout=dropout,
        )

        self.qkv = nn.Linear(q_dim, 3 * num_heads * self.head_dim, bias=bias)

        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(p=dropout),
        )

        self.o = nn.Linear(num_heads * self.head_dim, q_dim, bias=bias)

    def forward_qkv(self, tensor: Tensor, cache: Cache = None, **kwargs):
        qkv = self.qkv(tensor).unflatten(dim=-1, sizes=(3, self.num_heads, self.head_dim))
        q, k, v = qkv[..., 0, :, :], qkv[..., 1, :, :], qkv[..., 2, :, :]

        if cache is not None:
            ks, vs = cache
            k = torch.cat([ks, k], dim=-3)
            v = torch.cat([vs, v], dim=-3)

        return q, k, v


class CrossAttention(MultiAttentionMeta):
    def __init__(self, algo: Union[Type[dot], Type[detach]] = dot,
                 num_heads: int = 8, bias: bool = True, dropout: float = 0.0, *,
                 q_dim: int, kv_dim: int, **kwargs) -> None:
        super(CrossAttention, self).__init__(
            algo=algo, bias=bias, num_heads=num_heads,
            q_dim=q_dim, k_dim=kv_dim, v_dim=kv_dim,
            o_dim=q_dim, dropout=dropout,
        )

        self.q = nn.Linear(q_dim, num_heads * self.head_dim, bias=bias)
        self.kv = nn.Linear(kv_dim, 2 * num_heads * self.head_dim, bias=bias)

        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.Dropout(p=dropout),
        )

        self.o = nn.Linear(num_heads * self.head_dim, q_dim, bias=bias)

    def forward_qkv(self, q: Tensor, kv: Tensor, cache: Cache = None, **kwargs):
        q = self.q(q).unflatten(dim=-1, sizes=(self.num_heads, self.head_dim))

        if cache is not None:
            return q, *cache

        kv = self.kv(kv).unflatten(dim=-1, sizes=(2, self.num_heads, self.head_dim))
        return q, kv[..., 0, :, :], kv[..., 1, :, :]
