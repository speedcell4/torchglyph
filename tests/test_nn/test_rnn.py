import torch
from hypothesis import given, strategies as st
from torch import nn
from torchrua import pack_sequence

from tests.assertion import assert_close, assert_grad_close
from tests.strategy import sizes, TOKEN_SIZE, NUM_CONJUGATES, EMBEDDING_DIM, device, TINY_BATCH_SIZE
from torchglyph.nn.rnn import Lstm, LstmOrthogonalInit, LstmUniformInit


@given(
    cls=st.sampled_from([Lstm, LstmOrthogonalInit, LstmUniformInit]),
    token_sizes=sizes(TINY_BATCH_SIZE, TOKEN_SIZE),
    num_conjugates=sizes(NUM_CONJUGATES),
    input_size=sizes(EMBEDDING_DIM),
    hidden_size=sizes(EMBEDDING_DIM),
    bias=st.booleans(),
    bidirectional=st.booleans(),
)
def test_lstm(cls, token_sizes, num_conjugates, input_size, hidden_size, bias, bidirectional):
    actual_rnn = cls(
        num_conjugates=num_conjugates,
        input_size=input_size,
        hidden_size=hidden_size,
        bias=bias,
        bidirectional=bidirectional,
    ).to(device=device)

    excepted_rnn_list = [
        nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=bias,
            bidirectional=bidirectional,
        ).to(device=device) for _ in range(num_conjugates)
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

    inputs = [[
        torch.randn((token_size, input_size), device=device, requires_grad=True)
        for token_size in token_sizes
    ] for _ in range(num_conjugates)]

    actual = actual_rnn(pack_sequence([torch.stack(item, dim=1) for item in zip(*inputs)])).data
    excepted = torch.stack([
        excepted_rnn(pack_sequence(sequence))[0].data
        for excepted_rnn, sequence in zip(excepted_rnn_list, inputs)
    ], dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=[x for xs in inputs for x in xs])
