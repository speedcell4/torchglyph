from hypothesis import given, strategies as st

from torchglyph.vocab import Glove


@given(vec_dim=st.sampled_from([50, 100, 200]))
def test_glove(vec_dim: int) -> None:
    vectors = Glove(name='6B', dim=vec_dim)
    assert len(vectors) == 400000
    assert vectors.vec_dim == vec_dim
