import torch
from hypothesis import given, strategies as st
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from torchglyph.pipe import PaddedRawPtrPipe, PackedRawPtrPipe


def assert_pack_equivalent(lhs: PackedSequence, rhs: PackedSequence, rtol: float = 1e-5, atol: float = 1e-5):
    assert torch.allclose(lhs.data, rhs.data, rtol=rtol, atol=atol)
    assert torch.allclose(lhs.batch_sizes, rhs.batch_sizes, rtol=rtol, atol=atol)

    if lhs.sorted_indices is not None or rhs.sorted_indices is not None:
        assert torch.allclose(lhs.sorted_indices, rhs.sorted_indices, rtol=rtol, atol=atol)

    if lhs.unsorted_indices is not None or rhs.unsorted_indices is not None:
        assert torch.allclose(lhs.unsorted_indices, rhs.unsorted_indices, rtol=rtol, atol=atol)


@given(
    batch_size=st.integers(1, 20),
    vocab_size=st.integers(1, 100),
    max_seq_length=st.integers(2, 10),
)
def test_ptr_pipe(batch_size, vocab_size, max_seq_length):
    padded_data = torch.randint(0, vocab_size, (batch_size, max_seq_length))
    seq_length = torch.randint(1, max_seq_length, (batch_size,))
    packed_data = pack_padded_sequence(padded_data, seq_length, batch_first=True, enforce_sorted=False)

    x = [
        torch.randperm(l).tolist()
        for l in seq_length.tolist()
    ]

    p1 = PaddedRawPtrPipe(torch.device('cpu'), 0)
    p2 = PackedRawPtrPipe(torch.device('cpu'))

    y, _ = p1(x)
    y = pack_padded_sequence(padded_data.gather(dim=-1, index=y), seq_length, batch_first=True, enforce_sorted=False)

    z, _ = p2(x)
    z = z._replace(data=packed_data.data[z.data])

    assert_pack_equivalent(y, z)
