import torch
from hypothesis import given, strategies as st
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from torchglyph.functional import support_pack


@given(
    batch_sizes=st.lists(st.integers(1, 10), min_size=1, max_size=4),
    max_sentence_length=st.integers(2, 20),
    input_dim=st.integers(1, 20),
    output_dim=st.integers(1, 20),
)
def test_packing(batch_sizes, max_sentence_length, input_dim, output_dim):
    layer = support_pack(nn.Linear(input_dim, output_dim))

    x1 = torch.rand(*batch_sizes, input_dim)
    y1 = layer(x1)
    assert y1.size() == (*batch_sizes, output_dim)

    batch_size = sum(batch_sizes)
    x2 = pack_sequence([
        torch.randn((torch.randint(1, max_sentence_length, ()).item(), input_dim))
        for _ in range(batch_size)
    ], enforce_sorted=False)
    y2 = layer(x2)
    assert y2.size() == (batch_size, len(x2.batch_sizes), output_dim)
