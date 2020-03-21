from torchglyph.datasets import CoNLL2000Chunking, CoNLL2003NER
from torchglyph.datasets import SemEval2010T1NERCatalan, SemEval2010T1NERSpanish


def test_conll2000_chunking():
    train, test = CoNLL2000Chunking.new(batch_size=1, word_dim=None)
    assert len(train.dataset) == 8936
    assert len(test.dataset) == 2012


def test_conll2003_ner():
    train, dev, test = CoNLL2003NER.new(batch_size=1, word_dim=None)
    assert len(train.dataset) == 14987
    assert len(dev.dataset) == 3466
    assert len(test.dataset) == 3684


def test_semeval2010_catalan():
    train, dev, test = SemEval2010T1NERCatalan.new(batch_size=1, word_dim=None)
    assert len(train.dataset) == 8709
    assert len(dev.dataset) == 1445
    assert len(test.dataset) == 1698


def test_semeval2010_spanish():
    train, dev, test = SemEval2010T1NERSpanish.new(batch_size=1, word_dim=None)
    assert len(train.dataset) == 9022
    assert len(dev.dataset) == 1419
    assert len(test.dataset) == 1705
