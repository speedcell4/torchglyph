from einops import rearrange
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import PackedMeta, PackedSequential

from torchglyph.vocab import Vocab

__all__ = [
    'TokenEmbedding',
    'CharLstmEmbedding',
]


class TokenEmbedding(nn.Embedding, metaclass=PackedMeta):
    def __init__(self, embedding_dim: int, freeze: bool = False, *, vocab: Vocab = None,
                 num_embeddings: int = 0, padding_idx: int = None) -> None:
        if vocab is not None:
            super(TokenEmbedding, self).__init__(
                embedding_dim=embedding_dim,
                num_embeddings=len(vocab),
                padding_idx=vocab.pad_idx,
                _weight=vocab.vectors,
            )
        else:
            super(TokenEmbedding, self).__init__(
                embedding_dim=embedding_dim,
                num_embeddings=num_embeddings,
                padding_idx=padding_idx,
                _weight=None,
            )
        self.weight.requires_grad = not freeze

    def extra_repr(self) -> str:
        if not self.weight.requires_grad:
            return ', '.join([super(TokenEmbedding, self).extra_repr(), 'frozen'])
        return super(TokenEmbedding, self).extra_repr()

    def padding_mask(self, indices: Tensor) -> Tensor:
        return indices == (self.padding_idx if self.padding_idx is None else -100)


class CharLstmEmbedding(nn.Module):
    def __init__(self, hidden_dim: int = 50, num_layers: int = 1,
                 bias: bool = True, bidirectional: bool = True, dropout: float = 0.5, *,
                 char_embedding: TokenEmbedding) -> None:
        super(CharLstmEmbedding, self).__init__()

        self.embedding = PackedSequential(
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
