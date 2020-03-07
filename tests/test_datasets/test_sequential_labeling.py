from torchglyph.datasets.sequential_labeling import CoNLL2000Chunking, CoNLL2003NER


def test_conll2000_chunking() -> None:
    train, test = CoNLL2000Chunking.new(batch_size=10, word_dim=None)
    assert len(train.dataset) == 8936
    assert len(test.dataset) == 2012


def test_conll2003_ner() -> None:
    train, dev, test = CoNLL2003NER.new(batch_size=10, word_dim=None)
    assert len(train.dataset) == 14987
    assert len(dev.dataset) == 3466
    assert len(test.dataset) == 3684
