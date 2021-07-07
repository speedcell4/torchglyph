from einops.layers.torch import Reduce
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence
from torchrua import PackedMeta, PackedSequential

__all__ = [
    'TokenEmbedding',
    'CharLstmEmbedding',
]


class TokenEmbedding(nn.Embedding, metaclass=PackedMeta):
    pass


class CharLstmEmbedding(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 1,
                 bias: bool = True, bidirectional: bool = True,
                 dropout: float = 0.5, reduction: str = 'max', *,
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
        self.reduce = Reduce('(d l) b x -> b (d x)', reduction=reduction, d=self.num_directions)

    def forward(self, indices: PackedSequence) -> Tensor:
        embedding = self.embedding(indices)
        _, (encoding, _) = self.rnn(embedding)
        return self.reduce(encoding)
