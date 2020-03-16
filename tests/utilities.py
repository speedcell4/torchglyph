import torch
from torch.nn.utils.rnn import PackedSequence


def assert_pack_allclose(lhs: PackedSequence, rhs: PackedSequence, rtol: float = 1e-5, atol: float = 1e-5):
    assert torch.allclose(lhs.data, rhs.data, rtol=rtol, atol=atol)
    assert torch.allclose(lhs.batch_sizes, rhs.batch_sizes, rtol=rtol, atol=atol)

    if lhs.sorted_indices is not None or rhs.sorted_indices is not None:
        assert torch.allclose(lhs.sorted_indices, rhs.sorted_indices, rtol=rtol, atol=atol)

    if lhs.unsorted_indices is not None or rhs.unsorted_indices is not None:
        assert torch.allclose(lhs.unsorted_indices, rhs.unsorted_indices, rtol=rtol, atol=atol)
