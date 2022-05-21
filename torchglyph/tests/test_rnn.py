import torch
from hypothesis import given, strategies as st
from torch import nn
from torch.testing import assert_close
from torchrua import pack_sequence

from torchglyph.nn.rnn import LSTM


@given(
    token_sizes=st.lists(st.integers(1, 20), min_size=1, max_size=10),
    num_conjugates=st.integers(1, 5),
    input_size=st.integers(1, 50),
    hidden_size=st.integers(1, 50),
    bias=st.booleans(),
    bidirectional=st.booleans(),
)
def test_lstm(token_sizes, num_conjugates, input_size, hidden_size, bias, bidirectional):
    actual_rnn = LSTM(
        num_conjugates=num_conjugates,
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        bidirectional=bidirectional,
    )

    excepted_rnn_list = [
        nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            bidirectional=bidirectional,
        ) for _ in range(num_conjugates)
    ]

    with torch.no_grad():
        for index in range(num_conjugates):
            actual_rnn.weight_ih.data[index] = excepted_rnn_list[index].weight_ih_l0
            actual_rnn.weight_hh.data[index] = excepted_rnn_list[index].weight_hh_l0

            if bias:
                actual_rnn.bias_ih.data[index] = excepted_rnn_list[index].bias_ih_l0
                actual_rnn.bias_hh.data[index] = excepted_rnn_list[index].bias_hh_l0

            if bidirectional:
                actual_rnn.weight_ih_reverse.data[index] = excepted_rnn_list[index].weight_ih_l0_reverse
                actual_rnn.weight_hh_reverse.data[index] = excepted_rnn_list[index].weight_hh_l0_reverse

                if bias:
                    actual_rnn.bias_ih_reverse.data[index] = excepted_rnn_list[index].bias_ih_l0_reverse
                    actual_rnn.bias_hh_reverse.data[index] = excepted_rnn_list[index].bias_hh_l0_reverse

    sequence_list = [[
        torch.randn((token_size, input_size), requires_grad=True)
        for token_size in token_sizes
    ] for _ in range(num_conjugates)]

    actual = actual_rnn(pack_sequence([torch.stack(item, dim=1) for item in zip(*sequence_list)])).data
    excepted = torch.stack([
        excepted_rnn(pack_sequence(sequence))[0].data
        for excepted_rnn, sequence in zip(excepted_rnn_list, sequence_list)
    ], dim=1)

    assert_close(actual=actual, expected=excepted)
