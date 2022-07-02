from typing import List, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import init
from torch.nn.init import zeros_, ones_
from torch.nn.utils.rnn import PackedSequence
from torchrua import reverse_packed_indices

from torchglyph.functional import conjugated_linear
from torchglyph.nn.init import orthogonal_, xavier_uniform_

__all__ = [
    'Lstm',
    'LstmUniformInit',
    'LstmOrthogonalInit',
]


class Lstm(nn.Module):
    def __init__(self, num_conjugates: int, input_size: int, hidden_size: int,
                 bias: bool = True, bidirectional: bool = True) -> None:
        super(Lstm, self).__init__()

        self.num_conjugates = num_conjugates
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.bidirectional = bidirectional

        self.weight_ih = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4, input_size)))
        self.weight_hh = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4, hidden_size)))
        if bias:
            self.bias_ih = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4)))
            self.bias_hh = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4)))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        if bidirectional:
            self.weight_ih_reverse = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4, input_size)))
            self.weight_hh_reverse = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4, hidden_size)))
            if bias:
                self.bias_ih_reverse = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4)))
                self.bias_hh_reverse = nn.Parameter(torch.empty((num_conjugates, hidden_size * 4)))
            else:
                self.register_parameter('bias_ih_reverse', None)
                self.register_parameter('bias_hh_reverse', None)

        self.reset_parameters()

    def extra_repr(self) -> str:
        return ', '.join([
            f'num_conjugates={self.num_conjugates}',
            f'input_size={self.input_size}',
            f'hidden_size={self.hidden_size}',
            f'bias={self.bias}',
            f'bidirectional={self.bidirectional}',
        ])

    @torch.no_grad()
    def reset_parameters(self) -> None:
        std = self.hidden_size ** -0.5
        for weight in self.parameters():
            init.uniform_(weight, -std, std)

    def prepare_hx(self, sequence: PackedSequence) -> Tuple[List[Tensor], List[Tensor]]:
        hx = torch.zeros(
            (sequence.batch_sizes[0].item(), sequence.data.size()[1], self.hidden_size),
            dtype=sequence.data.dtype, device=sequence.data.device, requires_grad=False,
        )
        return [hx], [hx]

    def forward_cell(self, x: Tensor, h_prev: Tensor, c_prev: Tensor, weight_hh: Tensor) -> Tuple[Tensor, Tensor]:
        i, f, g, o = conjugated_linear(h_prev, weight=weight_hh, bias=x).chunk(4, dim=-1)

        c = torch.sigmoid(i) * torch.tanh(g) + torch.sigmoid(f) * c_prev
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

    def forward_unary(self, sequence: PackedSequence, weight_ih: Tensor, weight_hh: Tensor, bias: Tensor) -> Tensor:
        xs = conjugated_linear(sequence.data, weight=weight_ih, bias=bias)

        start, end = 0, 0
        hs, cs = self.prepare_hx(sequence=sequence)
        for batch_size in sequence.batch_sizes.detach().cpu().tolist():
            start, end = end, end + batch_size

            h, c = self.forward_cell(
                x=xs[start:end],
                h_prev=hs[-1][:batch_size],
                c_prev=cs[-1][:batch_size],
                weight_hh=weight_hh,
            )

            hs.append(h)
            cs.append(c)

        return torch.cat(hs[1:], dim=0)

    def forward(self, sequence: PackedSequence) -> PackedSequence:
        if not self.bidirectional:
            data = self.forward_unary(
                sequence=sequence,
                weight_ih=self.weight_ih,
                weight_hh=self.weight_hh,
                bias=(self.bias_ih + self.bias_hh) if self.bias else None,
            )

            return sequence._replace(data=data)
        else:
            indices = reverse_packed_indices(
                batch_sizes=sequence.batch_sizes,
                device=sequence.data.device,
            )

            data = self.forward_unary(
                sequence=sequence._replace(data=torch.cat([sequence.data, sequence.data[indices]], dim=1)),
                weight_ih=torch.cat([self.weight_ih, self.weight_ih_reverse], dim=0),
                weight_hh=torch.cat([self.weight_hh, self.weight_hh_reverse], dim=0),
                bias=torch.cat([self.bias_ih + self.bias_hh, self.bias_ih_reverse + self.bias_hh_reverse], dim=0)
                if self.bias else None,
            )

            data1, data2 = data.chunk(2, dim=1)
            return sequence._replace(data=torch.cat([data1, data2[indices]], dim=-1))


class LstmUniformInit(Lstm):
    pass


class LstmOrthogonalInit(Lstm):
    @torch.no_grad()
    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                xavier_uniform_(param, fan_in=self.input_size, fan_out=self.hidden_size * 4)
            elif 'weight_hh' in name:
                orthogonal_(self.weight_hh)
            elif 'bias_ih' in name or 'bias_hh' in name:
                zeros_(self.bias_ih)
                ones_(self.bias_ih[self.hidden_size:self.hidden_size * 2])
            else:
                raise KeyError(f'unknown parameter {name}')
