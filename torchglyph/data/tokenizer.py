import json
from logging import getLogger
from pathlib import Path
from typing import List

from tokenizers import Regex, decoders, normalizers, pre_tokenizers
from tokenizers.implementations import SentencePieceBPETokenizer, SentencePieceUnigramTokenizer

logger = getLogger(__name__)


def rglob(file: List[Path]):
    fs, ls = [], set()
    for path in file:
        for f in path.resolve().parent.rglob(path.name):  # type: Path
            if f.is_file():
                fs.append(str(f))

                *_, lang = f.name.strip().split('.')
                ls.add(lang)

    ls = sorted(list(ls))

    logger.info(f'files ({len(fs)}) => {json.dumps(fs, indent=2, ensure_ascii=False)}')
    logger.info(f'langs ({len(ls)}) => {ls}')
    return fs, ls


def normalizer() -> normalizers.Normalizer:
    return normalizers.Sequence([
        normalizers.Strip(),
        normalizers.Nmt(),
        normalizers.NFKC(),
        normalizers.Replace(Regex(r'\s+'), ' '),
    ])


def pre_tokenizer() -> pre_tokenizers.PreTokenizer:
    return pre_tokenizers.Sequence([
        # pre_tokenizers.Punctuation(), pre_tokenizers.Digits(),
        pre_tokenizers.Split(Regex(r' *(([\p{P}\p{S}])|(\d+))'), 'isolated'),
        pre_tokenizers.Metaspace(prepend_scheme='first'),
    ])


def bpe(punctuation: bool = True, unk_token: str = '<unk>') -> SentencePieceBPETokenizer:
    tokenizer = SentencePieceBPETokenizer(add_prefix_space=True, unk_token=unk_token)
    tokenizer.normalizer = normalizer()

    if punctuation:
        tokenizer.pre_tokenizer = pre_tokenizer()
        tokenizer.decoder = decoders.Metaspace(prepend_scheme='first')

    return tokenizer


def unigram(punctuation: bool = True, unk_token: str = '<unk>') -> SentencePieceUnigramTokenizer:
    tokenizer = SentencePieceUnigramTokenizer(add_prefix_space=True)
    tokenizer.normalizer = normalizer()

    if punctuation:
        tokenizer.pre_tokenizer = pre_tokenizer()
        tokenizer.decoder = decoders.Metaspace(prepend_scheme='first')

    return tokenizer
