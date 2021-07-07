import torch

from torchglyph.annotations import PackedSequence
from torchglyph.datasets.conll2003 import CoNLL2003


def test_conll2003():
    train, dev, test = CoNLL2003.new(batch_size=32, device=torch.device('cpu'))
    assert len(train.dataset) == 14987
    assert len(dev.dataset) == 3466
    assert len(test.dataset) == 3684

    for batch in train:
        assert isinstance(batch.word, PackedSequence)
        assert isinstance(batch.char, PackedSequence)
        assert isinstance(batch.tag, PackedSequence)

    for batch in dev:
        assert isinstance(batch.word, PackedSequence)
        assert isinstance(batch.char, PackedSequence)
        assert isinstance(batch.tag, PackedSequence)

    for batch in test:
        assert isinstance(batch.word, PackedSequence)
        assert isinstance(batch.char, PackedSequence)
        assert isinstance(batch.tag, PackedSequence)
