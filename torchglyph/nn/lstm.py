from typing import Type

from torch import nn
from torch.nn.utils.rnn import PackedSequence
from torchrua import RuaSequential


class LstmLayer(nn.Module):
    def __init__(self, hidden_size: int = 200, bidirectional: bool = True,
                 bias: bool = True, dropout: float = 0.5, *, input_size: int, is_last: bool) -> None:
        super(LstmLayer, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            bidirectional=bidirectional, bias=bias,
            dropout=0., num_layers=1,
        )
        self.output_size = self.rnn.hidden_size
        if self.rnn.bidirectional:
            self.output_size *= 2

        if is_last:
            self.layer_norm = nn.Identity()
        else:
            self.layer_norm = RuaSequential(
                nn.LayerNorm(self.output_size),
                nn.Dropout(dropout),
            )

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        encoding, _ = super(LstmLayer, self).forward(sequence)
        return self.layer_norm(encoding)


class LstmEncoder(nn.ModuleList):
    def __init__(self, layer: Type[LstmLayer] = LstmLayer, num_layers: int = 1, *, input_size: int) -> None:
        assert num_layers > 0

        modules = []
        for _ in range(1, num_layers):
            modules.append(layer(input_size=input_size, is_last=False))
            input_size = modules[-1].output_size
        modules.append(layer(input_size=input_size, is_last=True))

        super(LstmEncoder, self).__init__(modules=modules)
        self.output_size = modules[-1].output_size

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        for layer in self:
            sequence = layer(sequence)
        return sequence
