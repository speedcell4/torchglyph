import torch
from hypothesis import given, strategies as st
from torch import nn
from torch.testing import assert_close

from tests.assertion import assert_grad_close
from tests.strategy import device, sizes, BATCH_SIZE, NUM_CONJUGATES, EMBEDDING_DIM
from torchglyph.nn.linear import Linear


@given(
    batch_size=sizes(BATCH_SIZE),
    num_conjugates=sizes(NUM_CONJUGATES),
    input_size=sizes(EMBEDDING_DIM),
    hidden_size=sizes(EMBEDDING_DIM),
    bias=st.booleans()
)
def test_linear(batch_size, num_conjugates, input_size, hidden_size, bias):
    actual_linear = Linear(
        in_features=input_size, out_features=hidden_size,
        num_conjugates=num_conjugates, bias=bias,
    ).to(device=device)

    excepted_linears = [
        nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias).to(device=device)
        for _ in range(num_conjugates)
    ]

    with torch.no_grad():
        actual_linear.weight.data = torch.stack([linear.weight for linear in excepted_linears], dim=0)
        if bias:
            actual_linear.bias.data = torch.stack([linear.bias for linear in excepted_linears], dim=0)

    inputs = torch.randn((batch_size, num_conjugates, input_size), requires_grad=True, device=device)

    actual = actual_linear(inputs)
    excepted = torch.stack([
        linear(inputs[:, index, :])
        for index, linear in enumerate(excepted_linears)
    ], dim=1)

    assert_close(actual=actual, expected=excepted)
    assert_grad_close(actual=actual, expected=excepted, inputs=inputs)
