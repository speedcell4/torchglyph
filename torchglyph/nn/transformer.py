from typing import Optional

from torch import nn, Tensor

from torchglyph.nn.activation import Activations
from torchglyph.nn.attention import MultiHeadAttention

__all__ = [
    'TransformerFfn',
    'TransformerEncoderLayer',
    'TransformerDecoderLayer',
]


class TransformerFfn(nn.Sequential):
    def __init__(self, hidden_size: int, dropout: float,
                 activation: Activations = nn.Tanh,
                 bias: bool = True, *, in_size: int) -> None:
        super(TransformerFfn, self).__init__(
            nn.Linear(in_size, hidden_size, bias=bias),
            activation(),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(hidden_size, in_size, bias=bias),
        )

        self.in_size = in_size
        self.activation = activation.__name__
        self.dropout = dropout
        self.bias = bias

    def extra_repr(self) -> str:
        return ', '.join([
            f'{self.in_size}',
            f'dropout={self.dropout}',
            f'bias={self.bias}',
        ])

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}{self.activation}({self.extra_repr()})'


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 2048, num_heads: int = 8,
                 att_dropout: float = 0., ffn_dropout: float = 0., bias: bool = True, *,
                 in_size: int) -> None:
        super(TransformerEncoderLayer, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.ffn_dropout = ffn_dropout
        self.bias = bias

        self.att = MultiHeadAttention(
            num_heads=num_heads,
            head_dim=in_size // num_heads,
            dropout=att_dropout, bias=bias,
            q_dim=in_size, k_dim=in_size, v_dim=in_size,
        )
        self.ffn = TransformerFfn(
            dropout=ffn_dropout, bias=bias,
            in_size=in_size, hidden_size=hidden_size,
        )

        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        src = self.norm1(src + self.dropout(self.att(q=src, k=src, v=src, mask=src_mask)))
        src = self.norm2(src + self.dropout(self.ffn(src)))
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 2048, num_heads: int = 8,
                 att_dropout: float = 0., ffn_dropout: float = 0., bias: bool = True, *,
                 in_size: int) -> None:
        super(TransformerDecoderLayer, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.att_dropout = att_dropout
        self.ffn_dropout = ffn_dropout
        self.bias = bias

        self.tgt = MultiHeadAttention(
            num_heads=num_heads,
            head_dim=in_size // num_heads,
            dropout=att_dropout, bias=bias,
            q_dim=in_size, k_dim=in_size, v_dim=in_size,
        )
        self.src = MultiHeadAttention(
            num_heads=num_heads,
            head_dim=in_size // num_heads,
            dropout=att_dropout, bias=bias,
            q_dim=in_size, k_dim=in_size, v_dim=in_size,
        )
        self.ffn = TransformerFfn(
            dropout=ffn_dropout, bias=bias,
            in_size=in_size, hidden_size=hidden_size,
        )

        self.norm1 = nn.LayerNorm(in_size)
        self.norm2 = nn.LayerNorm(in_size)
        self.norm3 = nn.LayerNorm(in_size)
        self.dropout = nn.Dropout(ffn_dropout)

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        tgt = self.norm1(tgt + self.dropout(self.tgt(q=tgt, k=tgt, v=tgt, mask=tgt_mask)))
        tgt = self.norm2(tgt + self.dropout(self.src(q=tgt, k=src, v=src, mask=src_mask)))
        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt
