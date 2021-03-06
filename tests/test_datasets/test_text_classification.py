from torchglyph.datasets import AgNews


def test_agnews():
    train, test = AgNews.new(batch_size=1, word_dim=None, remove_missing=True)
    assert len(train) == 120000
    assert len(test) == 7600
