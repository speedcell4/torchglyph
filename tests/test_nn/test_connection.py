import torch
from hypothesis import given, strategies as st
from torch import nn

from torchglyph.nn.connection import ResNorm, DenseNorm, ReZero


@given(
    batch_sizes=st.lists(st.integers(1, 10), min_size=0, max_size=4),
    input_dim=st.integers(1, 20),
)
def test_resnorm_shape_grad(batch_sizes, input_dim):
    layer = ResNorm(input_dim=input_dim, sub_layer=nn.Linear(input_dim, input_dim))
    x = torch.rand((*batch_sizes, input_dim), requires_grad=True)
    y = layer(x)

    assert y.size() == (*batch_sizes, layer.output_dim)
    assert y.requires_grad


@given(
    batch_sizes=st.lists(st.integers(1, 10), min_size=0, max_size=4),
    input_dim=st.integers(1, 20),
)
def test_densenorm_shape_grad(batch_sizes, input_dim):
    layer = DenseNorm(input_dim=input_dim, sub_layer=nn.Linear(input_dim, input_dim))
    x = torch.rand((*batch_sizes, input_dim), requires_grad=True)
    y = layer(x)

    assert y.size() == (*batch_sizes, layer.output_dim)
    assert y.requires_grad


@given(
    batch_sizes=st.lists(st.integers(1, 10), min_size=0, max_size=4),
    input_dim=st.integers(1, 20),
)
def test_rezero_shape_grad(batch_sizes, input_dim):
    layer = ReZero(input_dim=input_dim, sub_layer=nn.Linear(input_dim, input_dim))
    x = torch.rand((*batch_sizes, input_dim), requires_grad=True)
    y = layer(x)

    assert y.size() == (*batch_sizes, layer.output_dim)
    assert y.requires_grad
