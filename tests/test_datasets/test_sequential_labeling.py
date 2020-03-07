from torchglyph.datasets.sequential_labeling import CoNLL2003


def test_conll2003() -> None:
    train, dev, test = CoNLL2003.new(batch_size=10, word_dim=None)
    assert len(train.dataset) == 14987
    assert len(dev.dataset) == 3466
    assert len(test.dataset) == 3684
