import torch

from torchglyph.datasets.conll2003 import CoNLL2003


def test_conll2003():
    train, dev, test = CoNLL2003.new(batch_size=32, device=torch.device('cpu'))
    assert len(train.dataset) == 14987
    assert len(dev.dataset) == 3466
    assert len(test.dataset) == 3684
