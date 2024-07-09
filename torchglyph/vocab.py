import itertools
from logging import getLogger
from typing import List, Tuple, Type

from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer
from torch.distributions.utils import lazy_property
from tqdm import tqdm

logger = getLogger(__name__)


def get_iterator(*datasets: Dataset, keys: List[str]):
    for dataset in tqdm(datasets):
        for data in tqdm(dataset):
            for key in keys:
                yield data[key]


class Vocab(object):
    Token: Type
    Index: Type
    registry = {}

    def __init__(self, vocab_size: int = 10_0000, min_freq: int = 0, *,
                 unk_token: str = None, pad_token: str = None,
                 bos_token: str = None, eos_token: str = None,
                 mask_token: str = None, special_tokens: Tuple[str, ...] = ()) -> None:
        super(Vocab, self).__init__()

        self._vocab_size = vocab_size
        self.min_freq = min_freq

        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.mask_token = mask_token

        special_tokens = [unk_token, pad_token, bos_token, eos_token, mask_token, *special_tokens]
        special_tokens = [token for token in special_tokens if token is not None]
        self.special_tokens = special_tokens

    @property
    def unk_id(self) -> int:
        return self.tokenizer.token_to_id(self.unk_token)

    @property
    def pad_id(self) -> int:
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def bos_id(self) -> int:
        return self.tokenizer.token_to_id(self.bos_token)

    @property
    def eos_id(self) -> int:
        return self.tokenizer.token_to_id(self.eos_token)

    @property
    def mask_id(self) -> int:
        return self.tokenizer.token_to_id(self.mask_token)

    def __init_subclass__(cls, **kwargs):
        if hasattr(cls, 'Token') and hasattr(cls, 'Index'):
            if (cls.__base__, cls.Token, cls.Index) in cls.registry:
                logger.warning(f'({cls.__base__}, {cls.Token}, {cls.Index}) is overwritten')
            cls.registry[cls.__base__, cls.Token, cls.Index] = cls

    def __class_getitem__(cls, tp) -> 'Vocab':
        return cls.registry[cls, tp[0], tp[1]]

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size(with_added_tokens=True)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'

    def extra_repr(self) -> str:
        return ', '.join([
            f'{len(self)}',
            *self.special_tokens,
        ])

    @lazy_property
    def tokenizer(self) -> Tokenizer:
        raise NotImplementedError

    @lazy_property
    def trainer(self):
        raise NotImplementedError

    def train_from_iterator(self, *iterators):
        return self.tokenizer.train_from_iterator(itertools.chain(*iterators), self.trainer)

    def encode(self, sequence: 'Token', pair: 'Token' = None, add_special_tokens: bool = True) -> 'Index':
        raise NotImplementedError

    def encode_batch(self, sequences: List['Token'], add_special_tokens: bool = True) -> List['Index']:
        raise NotImplementedError

    def inv(self, index: 'Index'):
        raise NotImplementedError

    def inv_batch(self, indices: List['Index']):
        raise NotImplementedError

    def decode(self, index: 'Index', skip_special_tokens: bool = False) -> str:
        raise NotImplementedError

    def decode_batch(self, indices: List['Index'], skip_special_tokens: bool = False) -> List[str]:
        raise NotImplementedError


class WordVocab(Vocab):
    @lazy_property
    def trainer(self) -> WordLevelTrainer:
        return WordLevelTrainer(
            show_progress=True,
            vocab_size=self._vocab_size,
            min_frequency=self.min_freq,
            special_tokens=self.special_tokens,
        )


class WordVocab00(WordVocab):
    Token = str
    Index = int

    @lazy_property
    def tokenizer(self) -> Tokenizer:
        obj = Tokenizer(model=models.WordLevel(unk_token=self.unk_token))
        return obj

    def encode(self, sequence: Token, pair: Token = None, add_special_tokens: bool = True) -> Index:
        encoding = self.tokenizer.encode(
            sequence, pair,
            is_pretokenized=False,
            add_special_tokens=add_special_tokens,
        )
        return encoding.ids[0]

    def encode_batch(self, sequences: List[Token], add_special_tokens: bool = True) -> List[Index]:
        encodings = self.tokenizer.encode_batch(
            sequences,
            is_pretokenized=False,
            add_special_tokens=add_special_tokens,
        )
        return [encoding.ids[0] for encoding in encodings]

    def inv(self, index: Index) -> str:
        return self.tokenizer.id_to_token(index)

    def inv_batch(self, index: List[Index]) -> List[str]:
        return [self.tokenizer.id_to_token(idx) for idx in index]

    def decode(self, index: Index, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode([index], skip_special_tokens=skip_special_tokens)

    def decode_batch(self, indices: List[Index], skip_special_tokens: bool = False) -> List[str]:
        return self.tokenizer.decode_batch(
            [[index] for index in indices],
            skip_special_tokens=skip_special_tokens,
        )


class WordVocab01(WordVocab):
    Token = str
    Index = List[int]

    @lazy_property
    def tokenizer(self) -> Tokenizer:
        obj = Tokenizer(model=models.WordLevel(unk_token=self.unk_token))
        obj.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.BertPreTokenizer(),
        ])
        return obj

    def encode(self, sequence: Token, pair: Token = None, add_special_tokens: bool = True) -> Index:
        encoding = self.tokenizer.encode(
            sequence, pair,
            is_pretokenized=False,
            add_special_tokens=add_special_tokens,
        )
        return encoding.ids

    def encode_batch(self, sequences: List[Token], add_special_tokens: bool = True) -> List[Index]:
        encodings = self.tokenizer.encode_batch(
            sequences,
            is_pretokenized=False,
            add_special_tokens=add_special_tokens,
        )
        return [encoding.ids for encoding in encodings]

    def inv(self, index: Index) -> List[str]:
        return [self.tokenizer.id_to_token(idx) for idx in index]

    def inv_batch(self, indices: List[Index]) -> List[List[str]]:
        return [[self.tokenizer.id_to_token(idx) for idx in index] for index in indices]

    def decode(self, index: Index, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(index, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, indices: List[Index], skip_special_tokens: bool = False) -> List[str]:
        return self.tokenizer.decode_batch(indices, skip_special_tokens=skip_special_tokens)


class WordVocab11(WordVocab):
    Token = List[str]
    Index = List[int]

    @lazy_property
    def tokenizer(self) -> Tokenizer:
        obj = Tokenizer(model=models.WordLevel(unk_token=self.unk_token))
        return obj

    def encode(self, sequence: Token, pair: Token = None, add_special_tokens: bool = True) -> Index:
        encoding = self.tokenizer.encode(
            sequence, pair,
            is_pretokenized=True,
            add_special_tokens=add_special_tokens,
        )
        return encoding.ids

    def encode_batch(self, sequences: List[Token], add_special_tokens: bool = True) -> List[Index]:
        encodings = self.tokenizer.encode_batch(
            sequences,
            is_pretokenized=True,
            add_special_tokens=add_special_tokens,
        )
        return [encoding.ids for encoding in encodings]

    def inv(self, index: Index) -> List[str]:
        return [self.tokenizer.id_to_token(idx) for idx in index]

    def inv_batch(self, indices: List[Index]) -> List[List[str]]:
        return [[self.tokenizer.id_to_token(idx) for idx in index] for index in indices]

    def decode(self, index: Index, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(index, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, indices: List[Index], skip_special_tokens: bool = False) -> List[str]:
        return self.tokenizer.decode_batch(indices, skip_special_tokens=skip_special_tokens)


class WordPieceVocab(Vocab):
    @lazy_property
    def trainer(self) -> WordPieceTrainer:
        return WordPieceTrainer(
            show_progress=True,
            vocab_size=self._vocab_size,
            min_frequency=self.min_freq,
            special_tokens=self.special_tokens,
        )


class WordPieceVocab01(WordPieceVocab):
    Token = str
    Index = List[int]

    @lazy_property
    def trainer(self) -> WordPieceTrainer:
        return WordPieceTrainer(
            show_progress=True,
            vocab_size=self._vocab_size,
            min_frequency=self.min_freq,
            special_tokens=self.special_tokens,
        )

    @lazy_property
    def tokenizer(self) -> Tokenizer:
        if self.unk_token is None:
            obj = Tokenizer(model=models.WordPiece())
        else:
            obj = Tokenizer(model=models.WordPiece(unk_token=self.unk_token))

        obj.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.UnicodeScripts(),
            pre_tokenizers.BertPreTokenizer(),
        ])
        return obj

    def encode(self, sequence: Token, pair: Token = None, add_special_tokens: bool = True) -> Index:
        encoding = self.tokenizer.encode(
            sequence, pair,
            is_pretokenized=False,
            add_special_tokens=add_special_tokens,
        )
        return encoding.ids

    def encode_batch(self, sequences: List[Token], add_special_tokens: bool = True) -> List[Index]:
        encodings = self.tokenizer.encode_batch(
            sequences,
            is_pretokenized=False,
            add_special_tokens=add_special_tokens,
        )
        return [encoding.ids for encoding in encodings]

    def inv(self, index: Index) -> List[str]:
        return [self.tokenizer.id_to_token(idx) for idx in index]

    def inv_batch(self, indices: List[Index]) -> List[List[str]]:
        return [[self.tokenizer.id_to_token(idx) for idx in index] for index in indices]

    def decode(self, index: Index, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(index, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, indices: List[Index], skip_special_tokens: bool = False) -> List[str]:
        return self.tokenizer.decode_batch(indices, skip_special_tokens=skip_special_tokens)
