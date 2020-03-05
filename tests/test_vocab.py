from torchglyph.vocab import Glove


def test_glove_repr() -> None:
    vectors = Glove('6B', 50)
    assert 'Glove(tok=400000, dim=50, unk_token=None)' == f'{vectors}'
    vectors.vectors = None
    assert 'Glove(tok=400000, unk_token=None)' == f'{vectors}'
