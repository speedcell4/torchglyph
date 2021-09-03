from typing import Optional, Type, Tuple, List

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

    def decode(self, tgt: Tensor,
               src_k: Tensor, src_v: Tensor,
               tgt_k: Tensor, tgt_v: Tensor,
               src_mask: Optional[Tensor] = None,
               tgt_mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        tgt, tgt_k, tgt_v = self.tgt.decode_tgt(q=tgt, k=tgt_k, v=tgt_v, mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt))

        tgt, src_k, src_v = self.src.decode_src(q=tgt, k=src_k, v=src_v, src_mask=src_mask)
        tgt = self.norm2(tgt + self.dropout(tgt))

        tgt = self.norm3(tgt + self.dropout(self.ffn(tgt)))
        return tgt, src_k, src_v, tgt_k, tgt_v


class Transformer(nn.Module):
    def __init__(self, num_enc_layers: int = 6, num_dec_layers: int = 6,
                 enc_layer_: Type[TransformerEncoderLayer] = TransformerEncoderLayer,
                 dec_layer_: Type[TransformerDecoderLayer] = TransformerDecoderLayer, *,
                 in_size: int) -> None:
        super(Transformer, self).__init__()

        self.encoder_layers = nn.ModuleList(modules=[
            enc_layer_(in_size=in_size) for _ in range(num_enc_layers)
        ])
        self.decoder_layers = nn.ModuleList(modules=[
            dec_layer_(in_size=in_size) for _ in range(num_dec_layers)
        ])

    def forward_encoder(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        for encoder_layer in self.encoder_layers:
            src = encoder_layer(src=src, src_mask=src_mask)
        return src

    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None,
                tgt_mask: Optional[Tensor] = None) -> Tensor:
        src = self.forward_encoder(src=src, src_mask=src_mask)

        for decoder_layer in self.decoder_layers:  # type: TransformerDecoderLayer
            tgt = decoder_layer.forward(
                src=src, src_mask=src_mask,
                tgt=tgt, tgt_mask=tgt_mask,
            )

        return tgt

    def decode(self, tgt: Tensor,
               src_k: List[Tensor], src_v: List[Tensor],
               tgt_k: List[Tensor], tgt_v: List[Tensor],
               src_mask: Optional[Tensor] = None,
               tgt_mask: Optional[Tensor] = None) -> \
            Tuple[Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:

        src_ks, src_vs, tgt_ks, tgt_vs = [], [], [], []
        for index, decoder_layer in enumerate(self.decoder_layers):  # type: (int, TransformerDecoderLayer)
            tgt, src_k_i, src_v_i, tgt_k_i, tgt_v_i = decoder_layer.decode(
                src_k=src_k[index], src_v=src_v[index], src_mask=src_mask,
                tgt_k=tgt_k[index], tgt_v=tgt_v[index], tgt_mask=tgt_mask, tgt=tgt,
            )
            src_ks.append(src_k_i)
            src_vs.append(src_v_i)
            tgt_ks.append(tgt_k_i)
            tgt_vs.append(tgt_v_i)

        return tgt, src_ks, src_vs, tgt_ks, tgt_vs
