from typing import List

from torchglyph.vocab import WordPieceVocab, WordVocab


def test_word_vocab00():
    vocab = WordVocab[str, int](unk_token='<unk>')
    vocab.train_from_iterator([
        'label1',
        'label2',
    ])

    index1 = vocab.encode('label1')
    index2 = vocab.encode('label2')
    index3 = vocab.encode('label3')

    assert vocab.inv(index1) == 'label1'
    assert vocab.inv(index2) == 'label2'
    assert vocab.inv(index3) == '<unk>'

    assert vocab.decode(index1) == 'label1'
    assert vocab.decode(index2) == 'label2'
    assert vocab.decode(index3) == '<unk>'

    index = vocab.encode_batch(['label1', 'label2', 'label3'])
    assert vocab.inv_batch(index) == ['label1', 'label2', '<unk>']
    assert vocab.decode_batch(index) == ['label1', 'label2', '<unk>']


def test_word_vocab01():
    vocab = WordVocab[str, List[int]](unk_token='<unk>')
    vocab.train_from_iterator([
        'this is the first sentence, and it is great',
        'another fantastic sentence here',
        # 'but this one is missing',
    ])

    index1 = vocab.encode('this is the first sentence, and it is great')
    index2 = vocab.encode('another fantastic sentence here')
    index3 = vocab.encode('but this one is missing')

    assert vocab.inv(index1) == ['this', 'is', 'the', 'first', 'sentence', ',', 'and', 'it', 'is', 'great']
    assert vocab.inv(index2) == ['another', 'fantastic', 'sentence', 'here']
    assert vocab.inv(index3) == ['<unk>', 'this', '<unk>', 'is', '<unk>']

    assert vocab.decode(index1) == 'this is the first sentence , and it is great'
    assert vocab.decode(index2) == 'another fantastic sentence here'
    assert vocab.decode(index3) == '<unk> this <unk> is <unk>'

    index = vocab.encode_batch([
        'this is the first sentence, and it is great',
        'another fantastic sentence here',
        'but this one is missing',
    ])
    assert vocab.inv_batch(index) == [
        ['this', 'is', 'the', 'first', 'sentence', ',', 'and', 'it', 'is', 'great'],
        ['another', 'fantastic', 'sentence', 'here'],
        ['<unk>', 'this', '<unk>', 'is', '<unk>'],
    ]
    assert vocab.decode_batch(index) == [
        'this is the first sentence , and it is great',
        'another fantastic sentence here',
        '<unk> this <unk> is <unk>',
    ]


def test_word_vocab11():
    vocab = WordVocab[List[str], List[int]](unk_token='<unk>')
    vocab.train_from_iterator([
        'this is the first sentence and it is great'.split(),
        'another fantastic sentence here'.split(),
        # 'but this one is missing'.split(),
    ])

    index1 = vocab.encode('this is the first sentence and it is great'.split())
    index2 = vocab.encode('another fantastic sentence here'.split())
    index3 = vocab.encode('but this one is missing'.split())

    assert vocab.inv(index1) == ['this', 'is', 'the', 'first', 'sentence', 'and', 'it', 'is', 'great']
    assert vocab.inv(index2) == ['another', 'fantastic', 'sentence', 'here']
    assert vocab.inv(index3) == ['<unk>', 'this', '<unk>', 'is', '<unk>']

    assert vocab.decode(index1) == 'this is the first sentence and it is great'
    assert vocab.decode(index2) == 'another fantastic sentence here'
    assert vocab.decode(index3) == '<unk> this <unk> is <unk>'

    index = vocab.encode_batch([
        'this is the first sentence and it is great'.split(),
        'another fantastic sentence here'.split(),
        'but this one is missing'.split(),
    ])
    assert vocab.inv_batch(index) == [
        ['this', 'is', 'the', 'first', 'sentence', 'and', 'it', 'is', 'great'],
        ['another', 'fantastic', 'sentence', 'here'],
        ['<unk>', 'this', '<unk>', 'is', '<unk>'],
    ]
    assert vocab.decode_batch(index) == [
        'this is the first sentence and it is great',
        'another fantastic sentence here',
        '<unk> this <unk> is <unk>',
    ]


def test_word_piece_vocab01_without_unk():
    vocab = WordPieceVocab[str, List[int]]()
    vocab.train_from_iterator([
        'this is the first sentence, and it is great',
        'another fantastic sentence here',
        # 'but this one is missing',
    ])

    index1 = vocab.encode('this is the first sentence, and it is great')
    index2 = vocab.encode('another fantastic sentence here')
    # index3 = vocab.encode('but this one is missing')

    assert vocab.inv(index1) == ['this', 'is', 'the', 'first', 'sentence', ',', 'and', 'it', 'is', 'great']
    assert vocab.inv(index2) == ['another', 'fantastic', 'sentence', 'here']
    # assert vocab.inv(index3) == ['<unk>', 'this', 'o', '##n', '##e', 'is', '<unk>']

    assert vocab.decode(index1) == 'this is the first sentence , and it is great'
    assert vocab.decode(index2) == 'another fantastic sentence here'
    # assert vocab.decode(index3) == '<unk> this o ##n ##e is <unk>'

    index = vocab.encode_batch([
        'this is the first sentence, and it is great',
        'another fantastic sentence here',
        # 'but this one is missing',
    ])
    assert vocab.inv_batch(index) == [
        ['this', 'is', 'the', 'first', 'sentence', ',', 'and', 'it', 'is', 'great'],
        ['another', 'fantastic', 'sentence', 'here'],
        # ['<unk>', 'this', 'o', '##n', '##e', 'is', '<unk>'],
    ]
    assert vocab.decode_batch(index) == [
        'this is the first sentence , and it is great',
        'another fantastic sentence here',
        # '<unk> this o ##n ##e is <unk>',
    ]
