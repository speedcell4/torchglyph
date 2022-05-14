import torch
from torch.nn.utils.rnn import PackedSequence

from torchglyph.datasets.named_entity_recognition import CoNLL2003


def test_conll2003():
    train, dev, test = CoNLL2003.new(batch_size=32, device=torch.device('cpu'))
    assert len(train.data_source) == 14987
    assert len(dev.data_source) == 3466
    assert len(test.data_source) == 3684

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
