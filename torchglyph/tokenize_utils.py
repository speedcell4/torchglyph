from typing import List, Tuple

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def get_iterator(*datasets, column_names: List[str]):
    for dataset in datasets:
        for row in dataset:
            for column_name in column_names:
                yield row[column_name]


def train_word_tokenizer(iterator, vocab_size: int = 40000, min_frequency: int = 1, pre_tokenizer: bool = True,
                         unk_token: str = '<unk>', special_tokens: Tuple[str, ...] = ()):
    tokenizer = Tokenizer(model=models.WordLevel(vocab=None, unk_token=unk_token))
    if pre_tokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), pre_tokenizers.Whitespace()])

    trainer = trainers.WordLevelTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=[t for t in [unk_token, *special_tokens] if t is not None],
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    return tokenizer


def train_word_piece_tokenizer(iterator, vocab_size: int = 40000, min_frequency: int = 1, pre_tokenizer: bool = True,
                               unk_token: str = '<unk>', special_tokens: Tuple[str, ...] = ()):
    tokenizer = Tokenizer(model=models.WordPiece(vocab=None, unk_token=unk_token))
    if pre_tokenizer:
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Punctuation(), pre_tokenizers.Whitespace()])
    tokenizer.decoder = decoders.WordPiece()

    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=[t for t in [unk_token, *special_tokens] if t is not None],
    )
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    return tokenizer


def tokenize(tokens: List[str], *, tokenizer: Tokenizer) -> List[str]:
    is_pretokenized = not isinstance(tokens, str)
    return tokenizer.encode(tokens, is_pretokenized=is_pretokenized, add_special_tokens=True).tokens


def tokenize_batch(batch: List[List[str]], *, tokenizer: Tokenizer) -> List[List[str]]:
    is_pretokenized = not isinstance(batch[0], str)
    return [s.tokens for s in tokenizer.encode_batch(batch, is_pretokenized=is_pretokenized, add_special_tokens=True)]


def encode(tokens: List[str], *, tokenizer: Tokenizer) -> List[int]:
    is_pretokenized = not isinstance(tokens, str)
    return tokenizer.encode(tokens, is_pretokenized=is_pretokenized, add_special_tokens=True).ids


def encode_batch(batch: List[List[str]], *, tokenizer: Tokenizer) -> List[List[int]]:
    is_pretokenized = not isinstance(batch[0], str)
    return [s.ids for s in tokenizer.encode_batch(batch, is_pretokenized=is_pretokenized, add_special_tokens=True)]


def segment(tokens: List[str], *, tokenizer: Tokenizer) -> Tuple[List[int], List[int]]:
    raise NotImplementedError


def segment_batch(tokens: List[List[str]], *, tokenizer: Tokenizer) -> Tuple[List[List[int]], List[List[int]]]:
    raise NotImplementedError


def decode(ids: List[int], *, tokenizer: Tokenizer) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False)


def decode_batch(sequences: List[List[int]], *, tokenizer: Tokenizer) -> List[str]:
    return tokenizer.decode_batch(sequences, skip_special_tokens=False)
