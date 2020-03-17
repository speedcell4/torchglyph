import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import PackedSequence


class ContextualLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_ctx_layers: int, num_rnn_layers: int,
                 bidirectional: bool = True, bias: bool = True) -> None:
        super(ContextualLSTM, self).__init__()

        self.ctx = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_ctx_layers, bias=bias,
            batch_first=False, bidirectional=bidirectional,
        )
        context_dim = self.ctx.hidden_size * (2 if self.ctx.bidirectional else 1)

        self.rnn = nn.LSTM(
            input_size=input_dim + context_dim,
            hidden_size=hidden_dim,
            num_layers=num_rnn_layers, bias=bias,
            batch_first=False, bidirectional=bidirectional,
        )

        self.encoding_dim = self.rnn.hidden_size * (2 if self.rnn.bidirectional else 1)

    def forward(self, inputs: PackedSequence, seq_ptr: PackedSequence) -> PackedSequence:
        _, (context, _) = self.ctx(inputs)
        context = rearrange(context, '(l d) b h -> l b (d h)', l=self.ctx.num_layers)

        contextual_embedding = inputs._replace(data=torch.cat([
            inputs.data, context[-1, seq_ptr.data],
        ], dim=-1))
        encoding, _ = self.rnn(contextual_embedding)
        return encoding
