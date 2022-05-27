import torch
from einops.layers.torch import Rearrange
from torch import Tensor
from torch import nn

from torchglyph.nn.linear import Linear


class MultiheadAttention(nn.Module):
    def __init__(self, q_dim: int, k_dim: int, v_dim: int, num_heads: int, head_dim: int, bias: bool) -> None:
        super(MultiheadAttention, self).__init__()

        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bias = bias
        self.tau = head_dim ** -0.5

        self.q = nn.Sequential(
            Linear(num_conjugates=num_heads, in_features=q_dim, out_features=head_dim, bias=bias),
            Rearrange('... q h x -> ... h q x'),
        )
        self.k = nn.Sequential(
            Linear(num_conjugates=num_heads, in_features=k_dim, out_features=head_dim, bias=bias),
            Rearrange('... k h x -> ... h x k'),
        )
        self.v = nn.Sequential(
            Linear(num_conjugates=num_heads, in_features=v_dim, out_features=head_dim, bias=bias),
            Rearrange('... k h y -> ... h k y'),
        )
        self.o = nn.Sequential(
            Rearrange('... h q y -> ... q () (h y)'),
            Linear(num_conjugates=num_heads, in_features=num_heads * head_dim, out_features=head_dim, bias=bias),
        )

    def extra_repr(self) -> str:
        return ', '.join([
            f'q={self.q_dim}',
            f'k={self.k_dim}',
            f'v={self.v_dim}',
            f'head_dim={self.head_dim}',
            f'num_heads={self.num_heads}',
            f'bias={self.bias}',
        ])

    def forward(self, q: Tensor, k: Tensor, v: Tensor, padding_mask: Tensor = None):
        q = self.q(q)
        k = self.k(k)
        v = self.v(v)

        scores = q @ k * self.tau
        if padding_mask is not None:
            scores, padding_mask = torch.broadcast_tensors(scores, padding_mask[..., None, :])
            scores = torch.masked_fill(scores, mask=padding_mask, value=-float('inf'))

        return self.o(torch.softmax(scores, dim=-1) @ v)
